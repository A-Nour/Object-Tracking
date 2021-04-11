classdef singleobjectracker
    %SINGLEOBJECTRACKER is a class containing functions to track a single
    %object in clutter. 
    %Model structures need to be called:
    %sensormodel: a structure specifies the sensor parameters
    %           P_D: object detection probability --- scalar
    %           lambda_c: average number of clutter measurements per time
    %           scan, Poisson distributed --- scalar 
    %           pdf_c: clutter (Poisson) density --- scalar
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
            %INITIATOR initializes singleobjectracker class
            %INPUT: density_class_handle: density class handle
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
            %         --- scalar 
            
            obj.density = density_class_handle;
            obj.gating.P_G = P_G;
            obj.gating.size = chi2inv(obj.gating.P_G,m_d);
            obj.reduction.w_min = log(w_min);
            obj.reduction.merging_threshold = merging_threshold;
            obj.reduction.M = M;
        end
        
        function estimates = nearestNeighbourFilter(obj, state, Z, sensormodel, motionmodel, measmodel)
            %NEARESTNEIGHBOURFILTER tracks a single object using nearest
            %neighbor association 
            %INPUT: state: a structure with two fields:
            %                x: object initial state mean --- (object state
            %                dimension) x 1 vector 
            %                P: object initial state covariance --- (object
            %                state dimension) x (object state dimension)
            %                matrix  
            %       Z: cell array of size (total tracking time, 1), each
            %       cell stores measurements of  
            %            size (measurement dimension) x (number of
            %            measurements at corresponding time step) 
            %OUTPUT:estimates: cell array of size (total tracking time, 1),
            %       each cell stores estimated object state of size (object
            %       state dimension) x 1   
            
            %     1- gating;
            %     2- calculate the predicted likelihood for each measurement in the gate;
            %     3- find the nearest neighbour measurement;
            %     4- compare the weight of the missed detection hypothesis and the weight of the object detection hypothesis created using the nearest neighbour measurement;
            %     5- if the object detection hypothesis using the nearest neighbour measurement has the highest weight, perform Kalman update;
            %     6- extract object state estimate;
            %     7- prediction.
            estimates = cell(size(Z, 1), 1);
            idx = 1;
            for z = Z'
                % 1- gating;
                [z_ingate, ~] = GaussianDensity.ellipsoidalGating(state, cell2mat(z), measmodel, obj.gating.size);

                % 2- calculate the predicted likelihood for each measurement in the gate;
                predict_likelihood = GaussianDensity.predictedLikelihood(state, z_ingate, measmodel);
                    
                if(size(z_ingate, 2) > 0)
                    % 3- find the nearest neighbour measurement;
                    [max_likelihood, max_idx] = max(predict_likelihood);
                    nearest_meas = z_ingate(:, max_idx);


                    % 4- compare the weight of the missed detection hypothesis 
                    % and the weight of the object detection hypothesis created 
                    % using the nearest neighbour measurement;
                    w_clutter = 1 - sensormodel.P_D;
                    w_det = sensormodel.P_D * exp(max_likelihood) / sensormodel.intensity_c;

                    % 5- update
                    if(w_det >= w_clutter)
                        state = GaussianDensity.update(state, nearest_meas, measmodel);
                    end
                end
                
                % 6- extract object state estimate;
                estimates(idx) = {state.x};
                
                % 7- prediction.
                state = GaussianDensity.predict(state, motionmodel);
                
                idx = idx + 1;
            end
        end
        
        
        function estimates = probDataAssocFilter(obj, state, Z, sensormodel, motionmodel, measmodel)
            %PROBDATAASSOCFILTER tracks a single object using probalistic
            %data association 
            %INPUT: state: a structure with two fields:
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
            %       state dimension) x 1  
            
            % 1- gating;
            % 2- create missed detection hypothesis;
            % 3- create object detection hypotheses for each detection inside the gate;
            % 4- normalise hypothesis weights;
            % 5- prune hypotheses with small weights, and then re-normalise the weights.
            % 6- merge different hypotheses using Gaussian moment matching;
            % 7- extract object state estimate;
            % 8- prediction.
            estimates = cell(size(Z, 1), 1);
            est_idx = 1;
            for z = Z'
                % 1- gating;
                [z_ingate, ~] = GaussianDensity.ellipsoidalGating(state, cell2mat(z), measmodel, obj.gating.size);
                weights = zeros(size(z_ingate,2)+1, 1);
                hypotheses = repmat(struct('x',[],'P',eye(size(state.P,1))), size(z_ingate, 2)+1, 1);
                
                if(size(z_ingate, 2) > 0)
                    % 2- create missed detection hypothesis;
                    weights(1, 1) = log(1 - sensormodel.P_D);
                    hypotheses(1) = state;
                    
                    % 3- create object detection hypotheses for each detection inside the gate;
                    predict_likelihood = GaussianDensity.predictedLikelihood(state, z_ingate, measmodel);
                    %likelihood = arrayfun(@(likelihood)exp(likelihood), predict_likelihood);
                    weights(2:size(weights,1), 1) = predict_likelihood + log(sensormodel.P_D / sensormodel.intensity_c)*ones(size(predict_likelihood,1), 1);
                    
                    for idx = 2 : size(weights,1)
                        hypotheses(idx) = GaussianDensity.update(state, z_ingate(:, idx-1), measmodel);
                    end
                    % 4- normalise hypothesis weights;
                    [weights, ~] = normalizeLogWeights(weights);
                    
                    % 5- prune hypotheses with small weights, and then re-normalise the weights.
                    hypotheses(weights < obj.reduction.w_min) = [];
                    weights(weights < obj.reduction.w_min) = [];
                    [weights, ~] = normalizeLogWeights(weights);
                    
                    if(size(weights,1) ~= 0)
                        % 6- merge different hypotheses using Gaussian moment matching;
                        state = GaussianDensity.momentMatching(weights, hypotheses);
                    end
                end
                
                % 7- extract object state estimate;
                estimates(est_idx) = {state.x};
                est_idx = est_idx + 1;
                
                % 8- prediction.
                state = GaussianDensity.predict(state, motionmodel);
            end
            
        end
        
        function estimates = GaussianSumFilter(obj, state, Z, sensormodel, motionmodel, measmodel)
            %GAUSSIANSUMFILTER tracks a single object using Gaussian sum
            %filtering
            %INPUT: state: a structure with two fields:
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
            %       state dimension) x 1  
 
            % for each hypothesis, create missed detection hypothesis;
            % for each hypothesis, perform ellipsoidal gating and only create object detection hypotheses for detections inside the gate;
            % normalise hypothsis weights;
            % prune hypotheses with small weights, and then re-normalise the weights;
            % hypothesis merging (to achieve this, you just need to directly call function hypothesisReduction.merge.);
            % cap the number of the hypotheses, and then re-normalise the weights;
            % extract object state estimate using the most probably hypothesis estimation;
            % for each hypothesis, perform prediction.
            
            estimates = cell(size(Z, 1), 1);
            hypotheses = state;
            est_idx = 1;
            tmp_weights = zeros(1, 0);
            tmp_hypotheses = repmat(struct('x',[],'P',eye(size(state.P,1))), 1, 0);
            
            for z = Z'    
                % for each hypothesis, perform ellipsoidal gating and only 
                % create object detection hypotheses for detections inside the gate;  
                for hyp = hypotheses'
                    % for each hypothesis, perform ellipsoidal gating
                    [z_ingate, ~] = GaussianDensity.ellipsoidalGating(hyp, cell2mat(z), measmodel, obj.gating.size);
                    
                    local_weights = zeros(1, 1);
                    local_hypotheses = repmat(struct('x',[],'P',eye(size(state.P,1))), 1, 1);
                
                    % for each hypothesis, create missed detection hypothesis;
                    local_weights(1, 1) = log(1 - sensormodel.P_D);
                    local_hypotheses(1, 1) = hyp;
                                        
                    if(size(z_ingate, 2) > 0)
                       
                        %create object detection hypotheses for each detection inside the gate;
                        predict_likelihood = GaussianDensity.predictedLikelihood(hyp, z_ingate, measmodel);
                        local_weights(2:size(predict_likelihood,1) + 1, 1) = predict_likelihood + log(sensormodel.P_D / sensormodel.intensity_c)*ones(size(predict_likelihood,1), 1);
                        
                        idx = 2;
                        for z_i = z_ingate
                            local_hypotheses(1, idx) = GaussianDensity.update(hyp, z_i, measmodel);
                            idx = idx + 1;
                        end
                    end 
                    
                    n = size(tmp_hypotheses,2) + 1;
                    m = n + size(local_hypotheses, 2) - 1;
                    tmp_hypotheses(1, n:m) = local_hypotheses;
                    tmp_weights(n:m, 1) = local_weights;
                end
                
                hypotheses = tmp_hypotheses;
                weights = tmp_weights;
                
                % normalise hypothsis weights;
                [weights, ~] = normalizeLogWeights(weights);

                tmp_weights = zeros(1, 0);
                tmp_hypotheses = repmat(struct('x',[],'P',eye(size(state.P,1))), 1, 0);

                % prune hypotheses with small weights, and then re-normalise the weights;
                [weights,hypotheses] = hypothesisReduction.prune(weights,...
                    hypotheses, obj.reduction.w_min);
                [weights, ~] = normalizeLogWeights(weights);
                
                % hypothesis merging (to achieve this, you just need to 
                % directly call function hypothesisReduction.merge.);
                [weights,hypotheses] = hypothesisReduction.merge(weights,...
                    hypotheses,obj.reduction.merging_threshold,obj.density);
                [weights, ~] = normalizeLogWeights(weights);
                
                % cap the number of the hypotheses, and then re-normalise the weights;
                [weights, hypotheses] = hypothesisReduction.cap(weights, hypotheses, obj.reduction.M);                
                [weights, ~] = normalizeLogWeights(weights);
                
                % extract object state estimate using the most probably hypothesis estimation;
                [~, out_idx] = max(weights);
                
                %extract object state estimate;
                estimates(est_idx) = {hypotheses(out_idx).x};
                est_idx = est_idx + 1;
                
                %predict each local hypothesis.
                hypotheses = arrayfun(@(hyp) GaussianDensity.predict(hyp, motionmodel), hypotheses); 
            end
        end
    end
end

