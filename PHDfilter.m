classdef PHDfilter
    %PHDFILTER is a class containing necessary functions to implement the
    %PHD filter 
    %Model structures need to be called:
    %    sensormodel: a structure which specifies the sensor parameters
    %           P_D: object detection probability --- scalar
    %           lambda_c: average number of clutter measurements per time scan, 
    %                     Poisson distributed --- scalar
    %           pdf_c: value of clutter pdf --- scalar
    %           intensity_c: Poisson clutter intensity --- scalar
    %       motionmodel: a structure which specifies the motion model parameters
    %           d: object state dimension --- scalar
    %           F: function handle return transition/Jacobian matrix
    %           f: function handle return predicted object state
    %           Q: motion noise covariance matrix
    %       measmodel: a structure which specifies the measurement model parameters
    %           d: measurement dimension --- scalar
    %           H: function handle return transition/Jacobian matrix
    %           h: function handle return the observation of the target state
    %           R: measurement noise covariance matrix
    %       birthmodel: a structure array which specifies the birth model (Gaussian
    %       mixture density) parameters --- (1 x number of birth components)
    %           w: weights of mixture components (in logarithm domain)
    %           x: mean of mixture components
    %           P: covariance of mixture components
    
    properties
        density %density class handle
        paras   %parameters specify a PPP
    end
    
    methods
        function obj = initialize(obj,density_class_handle,birthmodel)
            %INITIATOR initializes PHDfilter class
            %INPUT: density_class_handle: density class handle
            %OUTPUT:obj.density: density class handle
            %       obj.paras.w: weights of mixture components --- vector
            %                    of size (number of mixture components x 1)
            %       obj.paras.states: parameters of mixture components ---
            %                    struct array of size (number of mixture
            %                    components x 1) 
            
            obj.density = density_class_handle;
            obj.paras.w = [birthmodel.w]';
            obj.paras.states = rmfield(birthmodel,'w')';
        end
        
        function obj = predict(obj,motionmodel,P_S,birthmodel)
            %PREDICT performs PPP prediction step
            %INPUT: P_S: object survival probability
            n_survive = size(obj.paras.w, 1);
            n_birth = size(birthmodel, 2);
            
            % reserve memory for hypotheses
            hypotheses = repmat(struct('w',0,'x',[],'P',eye(motionmodel.d)),[1,n_survive + n_birth]);
            
            % birth obj processing
            for h = 1 : n_birth
                hypotheses(h) = birthmodel(h);
            end
            
            % survival obj processing
            for h = 1 : n_survive
                F = motionmodel.F(obj.paras.states(h).x);
                hypotheses(h+n_birth).w = log(P_S) + obj.paras.w(h);
                hypotheses(h+n_birth).x = motionmodel.f(obj.paras.states(h).x);
                hypotheses(h+n_birth).P = F*obj.paras.states(h).P*transpose(F) + motionmodel.Q;
            end
            
            obj.paras.w = [hypotheses.w]';
            obj.paras.states = rmfield(hypotheses,'w')';
        end
        
        function obj = update(obj,z,measmodel,intensity_c,P_D,gating)
            %UPDATE performs PPP update step and PPP approximation
            %INPUT: z: measurements --- matrix of size (measurement dimension 
            %          x number of measurements)
            %       intensity_c: Poisson clutter intensity --- scalar
            %       P_D: object detection probability --- scalar
            %       gating: a struct with two fields: P_G, size, used to
            %               specify the gating parameters
            H_old = size(obj.paras.w, 1);
            mk = size(z, 2);

            used_meas_d = false(mk,1);
            gating_matrix_d = false(mk,H_old);
            for i = 1:H_old
                %Perform gating
                [~,gating_matrix_d(:,i)] = ...
                    obj.density.ellipsoidalGating(obj.paras.states(i),z,measmodel,gating.size);
                used_meas_d = used_meas_d | gating_matrix_d(:,i);
            end
            
            mh = sum(sum(gating_matrix_d)); 
            %z_d = z(:,used_meas_d);
            z_d_idx = find(used_meas_d);
            
            
            % reserve memory for hypotheses
            hypotheses = repmat(struct('w',0,'x',[],'P',eye(size(obj.paras.states(1).P, 1))),[1,H_old + mh]);
            
            % update ppp part "undetected objects"
            for h = 1 : H_old
                hypotheses(h).w = log(1-P_D) + obj.paras.w(h);
                hypotheses(h).x = obj.paras.states(h).x;
                hypotheses(h).P = obj.paras.states(h).P;
            end            
            
            % update MB part "detected objects"
            % 1- update params "constant for all hyp for each measurment"
            update_params = repmat(struct('z',[],'S',eye(size(measmodel.R,1)),'K',[], 'P', eye(size(obj.paras.states(1).P,1))),[1,H_old]);
            for h = 1 : H_old
                H_k = measmodel.H(obj.paras.states(h).x);
                update_params(h).z = measmodel.h(obj.paras.states(h).x);
                update_params(h).S = measmodel.R + H_k*obj.paras.states(h).P*transpose(H_k);
                update_params(h).S = (update_params(h).S + update_params(h).S')/2;
                update_params(h).K = (obj.paras.states(h).P*transpose(H_k))/update_params(h).S;
                update_params(h).P = (eye(size(obj.paras.states(h).x, 1)) - update_params(h).K*H_k) * obj.paras.states(h).P;
            end
            
            % 2 - update using KF and normalize weights
            hyp_idx = H_old + 1;
            for z_idx = z_d_idx'
               z_d = z(:, z_idx);
               o_d_idx = find(gating_matrix_d(z_idx, :));
               sum_w = 0;
               start_idx = hyp_idx;
               
               for o_idx = o_d_idx
                    hypotheses(hyp_idx).x = obj.paras.states(o_idx).x + ...
                        update_params(o_idx).K*(z_d - update_params(o_idx).z);

                    hypotheses(hyp_idx).P = update_params(o_idx).P;

                    hypotheses(hyp_idx).w = log(P_D) + obj.paras.w(o_idx) + ...
                        obj.density.predictedLikelihood(obj.paras.states(o_idx) ,z_d ,measmodel);

                    sum_w = sum_w + exp(hypotheses(hyp_idx).w);

                    hyp_idx = hyp_idx + 1;
               end
               
                w_norm = log(sum_w + intensity_c);
                for o_idx = o_d_idx
                        hypotheses(start_idx).w = hypotheses(start_idx).w - w_norm;
                        start_idx = start_idx + 1;
                end 
            end    
            
            obj.paras.w = [hypotheses.w]';
            obj.paras.states = rmfield(hypotheses,'w')';
        end
        
        function obj = componentReduction(obj,reduction)
            %COMPONENTREDUCTION approximates the PPP by representing its
            %intensity with fewer parameters
            
            %Pruning
            [obj.paras.w, obj.paras.states] = hypothesisReduction.prune(obj.paras.w, obj.paras.states, reduction.w_min);
            %Merging
            if length(obj.paras.w) > 1
                [obj.paras.w, obj.paras.states] = hypothesisReduction.merge(obj.paras.w, obj.paras.states, reduction.merging_threshold, obj.density);
            end
            %Capping
            [obj.paras.w, obj.paras.states] = hypothesisReduction.cap(obj.paras.w, obj.paras.states, reduction.M);
        end
        
        function estimates = PHD_estimator(obj)
            %PHD_ESTIMATOR performs object state estimation in the GMPHD filter
            %OUTPUT:estimates: estimated object states in matrix form of
            %                  size (object state dimension) x (number of
            %                  objects) 
            n = min(round(sum(exp(obj.paras.w))), size(obj.paras.w,1));
            [~, ind] = maxk(obj.paras.w, n); 
            estimates = [obj.paras.states(ind).x];
        end
        
    end
    
end