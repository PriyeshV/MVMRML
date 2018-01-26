function [ predictions] = get_prediction_views(Iteration)
    
    n_views = size(Iteration(1).view,2);
    [~,n_labels] = size(Iteration(1).view(1).predicted_labels);
    n_unlabelled = size(Iteration(1).view(1).classifier(1).PosterioriPos,1);
    predictions = zeros(n_unlabelled,n_labels);    
    
    for label_id = 1:n_labels        
        poscfd = zeros(n_unlabelled,1);
        negcfd = zeros(n_unlabelled,1);   
        
        for view_id = 1:n_views
            poscfd = poscfd + Iteration(1).view(view_id).classifier(label_id).PosterioriPos;
            negcfd = negcfd + (1 - Iteration(1).view(view_id).classifier(label_id).PosterioriPos);
        end        
        predictions(:,label_id) = poscfd >= negcfd;
    end
    
end

