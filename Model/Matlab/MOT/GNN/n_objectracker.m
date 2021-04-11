classdef n_objectracker
    %N_OBJECTRACKER is a class containing functions to track n object in
    %clutter. 
    %Model structures need to be called:
    %sensormodel: a structure specifies the sensor parameters
    %           P_D: object detection probability --- scalar
    %           lambda_c: average number of clutter measurements per time
    %           scan, Poisson distributed --- scalar 
    %           pdf_c: clutter (Poisson) intensity --- scalar
    %           intensity_c: clutter (Poisson) intensity --- scalar
    %motionmodel: a structure specifies the motion model parameters
    %           d: object state dimension --- scalar
    %           F: function handle return transition/Jacobian matrix
    %           f: function handle return predicted object state
    %           Q: motion noise covariance matrix
    %measmodel: a structure specifies the measurement model parameters
    %           d: measurement dimension --- scalar
    %           H: function handle return transition/Jacobian matrix
    %           h: function handle return the observation of the object
    %           state 
    %           R: measurement noise covariance matrix
    
    properties
        gating      %specify gating parameter
        reduction   %specify hypothesis reduction parameter
        density     %density class handle
    end
    
    methods
        
        function obj = initialize(obj,density_class_handle,P_G,m_d,w_min,merging_threshold,M)
            %INITIATOR initializes n_objectracker class
            %INPUT: density_class_handle: density class handle
            %       P_D: object detection probability
            %       P_G: gating size in decimal --- scalar
            %       m_d: measurement dimension --- scalar
            %       wmin: allowed minimum hypothesis weight --- scalar
            %       merging_threshold: merging threshold --- scalar
            %       M: allowed maximum number of hypotheses --- scalar
            %OUTPUT:  obj.density: density class handle
            %         obj.gating.P_G: gating size in decimal --- scalar
            %         obj.gating.size: gating size --- scalar
            %         obj.reduction.w_min: allowed minimum hypothesis
            %         weight in logarithmic scale --- scalar 
            %         obj.reduction.merging_threshold: merging threshold
            %         --- scalar 
            %         obj.reduction.M: allowed maximum number of hypotheses
            %         used in TOMHT --- scalar 
            obj.density = density_class_handle;
            obj.gating.P_G = P_G;
            obj.gating.size = chi2inv(obj.gating.P_G,m_d);
            obj.reduction.w_min = log(w_min);
            obj.reduction.merging_threshold = merging_threshold;
            obj.reduction.M = M;
        end
        
                function estimates = GNNfilter(obj, states, Z, sensormodel, motionmodel, measmodel)
            %GNNFILTER tracks n object using global nearest neighbor
            %association 
            %INPUT: obj: an instantiation of n_objectracker class
            %       states: structure array of size (1, number of objects)
            %       with two fields: 
            %                x: object initial state mean --- (object state
            %                dimension) x 1 vector 
            %                P: object initial state covariance --- (object
            %                state dimension) x (object state dimension)
            %                matrix  
            %       Z: cell array of size (total tracking time, 1), each
            %       cell stores measurements of size (measurement
            %       dimension) x (number of measurements at corresponding
            %       time step)  
            %OUTPUT:estimates: cell array of size (total tracking time, 1),
            %       each cell stores estimated object state of size (object
            %       state dimension) x (number of objects)
            
            
            %     implement ellipsoidal gating for each predicted local hypothesis seperately, see Note below for details;
            %     construct 2D cost matrix of size (number of objects, number of measurements that at least fall inside the gates + number of objects);
            %     find the best assignment matrix using a 2D assignment solver;
            %     create new local hypotheses according to the best assignment matrix obtained;
            %     extract object state estimates;
            %     predict each local hypothesis.
            
            estimates = cell(size(Z, 1), 1);
            timeIdx = 1;
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Gating and creating score matrix %
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            l_u = log(1 - sensormodel.P_D);
            tmp = eye(size(states, 2)) * -l_u;
            tmp(tmp == 0) = Inf;
                
            for measurments_t = Z'
                scoreMtx = ones(size(states, 2), size(cell2mat(measurments_t), 2) + size(states, 2)).*Inf;
                state_idx = 1;
                
                for state = states
                    [z_ingate, meas_in_gate] = GaussianDensity.ellipsoidalGating(state, cell2mat(measurments_t), measmodel, obj.gating.size);
                    l_d = GaussianDensity.predictedLikelihood(state, z_ingate, measmodel);
                    w_det = log(sensormodel.P_D) + l_d - log(sensormodel.intensity_c);
                    scoreMtx(state_idx, find(meas_in_gate)) = -w_det;
                    state_idx = state_idx + 1;
                end
                
                scoreMtx(:, size(cell2mat(measurments_t), 2)+1: + size(scoreMtx, 2)) = tmp;
                
                % remove columns with only Inf enrtries
                removedCol = find(all(scoreMtx(:,:) == Inf));
                filteredMeas = cell2mat(measurments_t);
                scoreMtx(:, removedCol)=[];     
                filteredMeas(:, removedCol)=[];
                
                if(size(filteredMeas, 2) ~= 0)
                    %find the best assignment matrix using a 2D assignment solver;
                    [col4row,~,~] = assign2D(scoreMtx);
                    
                    %create new local hypotheses according to the best assignment matrix obtained;
                    estIdx = 1;
                    for measIdx = col4row'
                        if(measIdx <= size(filteredMeas, 2))
                            states(estIdx) = GaussianDensity.update(states(estIdx), filteredMeas(:, measIdx), measmodel);
                        end
                        estIdx = estIdx + 1;
                    end
                    
                end
                
                %extract object state estimates;
                estimates(timeIdx) = {[states.x]};
                timeIdx = timeIdx + 1;

                %predict each local hypothesis.
                states = arrayfun(@(state) GaussianDensity.predict(state, motionmodel), states);               
            end
        end
    end
end

