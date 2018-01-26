function [ prediction_1, prediction_2,prediction_a] = get_Predictions_CoTraining(unlabelled_indices,view,link,v_Iteration,l_Iteration,Base_a,Base_r,v_mode,labels)
    %returns final MVMRML predictions
    %prediction_1 is obtained from average of probabilities of MV and MR model
    %prediction_2 is obtained from product of probabilities of MV and MR model
    %prediction_a is obtained from MV
    %Base_a is the initial supervised MV model
    %Base_r is the initial supervised MR model
    %Base_a and Base_r are used to bootstrap label correlations
    
    n_unlabelled = nnz(unlabelled_indices);
    [~,n_labels] = size(v_Iteration(1).view(1).predicted_labels);
    l_prediction_Pos = ones(n_unlabelled,n_labels);
    l_prediction_Neg = ones(n_unlabelled,n_labels);
    a_prediction_Pos = ones(n_unlabelled,n_labels);
    a_prediction_Neg = ones(n_unlabelled,n_labels);
    n_views = size(view,2);
    n_links = size(link,2);
    prediction_1 = zeros(n_unlabelled,n_labels);
    prediction_2 = zeros(n_unlabelled,n_labels);
    
    for view_id = 1:n_views
        for label_id = 1:n_labels
            [PosterioriPos,PosterioriNeg] = BinaryClassify_test(Base_a.view(view_id).classifier(label_id).model, [view(view_id).r_view(unlabelled_indices,:) zeros(n_unlabelled,n_labels-1)]); 
            a_prediction_Pos(:,label_id) = a_prediction_Pos(:,label_id) .* PosterioriPos;
            a_prediction_Neg(:,label_id) = a_prediction_Neg(:,label_id) .* PosterioriNeg;
        end        
    end
    prediction_a = a_prediction_Pos >= a_prediction_Neg;
    labels(unlabelled_indices,:) = prediction_a;
    
    for link_id = 1:n_links
        tmp_view = prepare_link_view(link(link_id).links,labels);
        for label_id = 1:n_labels
                link(link_id).view(1,label_id).value = tmp_view;
        end
    end            
    clear tmp_view
    
    for link_id = 1:n_links
        for label_id = 1:n_labels
            [PosterioriPos,PosterioriNeg] = BinaryClassify_test(Base_r.view(link_id).classifier(label_id).model, [link(link_id).r_view(unlabelled_indices,:) link(link_id).view(label_id).value(unlabelled_indices,:)]); 
            l_prediction_Pos(:,label_id) = l_prediction_Pos(:,label_id) .* PosterioriPos;
            l_prediction_Neg(:,label_id) = l_prediction_Neg(:,label_id) .* PosterioriNeg;
        end
    end
    prediction_r = l_prediction_Pos >= l_prediction_Neg;
    
    for label_id = 1:n_labels
        PosterioriPos = a_prediction_Pos(:,label_id) + l_prediction_Pos(:,label_id);
        PosterioriNeg = a_prediction_Neg(:,label_id) + l_prediction_Neg(:,label_id);
        prediction_1(:,label_id) = PosterioriPos >= PosterioriNeg;
    end
    
    for label_id = 1:n_labels
        PosterioriPos = a_prediction_Pos(:,label_id) .* l_prediction_Pos(:,label_id);
        PosterioriNeg = a_prediction_Neg(:,label_id) .* l_prediction_Neg(:,label_id);
        prediction_2(:,label_id) = PosterioriPos >= PosterioriNeg;
    end
    
   %======================

   labels(unlabelled_indices,:) = prediction_2;
    %UPDATE ATTRIBUTE VIEW'S LABEL CORRELATIONS and LINK PREDICTION
    for label_id = 1:n_labels
       for view_id = 1:n_views   
            label_correlations = prepare_vmode(labels,label_id);
            label_correlations(label_correlations == 0) = -1;
            label_correlations = 1 - label_correlations;
            if v_mode == 2
                   view(1,view_id).view(1,label_id).value(:,end-(n_labels-2):end) = label_correlations;
            elseif v_mode == 4
                 view(1,view_id).view(1,label_id).value(:,(end-2*n_labels)+2:end-n_labels) = label_correlations;
                 view(1,view_id).view(1,label_id).value(:,end-n_labels+1:end) = prediction_r;                                                       
            end
       end
    end    
    
    for view_id = 1:n_views
        for label_id = 1:n_labels
            [PosterioriPos,PosterioriNeg] = BinaryClassify_test(v_Iteration.view(view_id).classifier(label_id).model, [view(view_id).r_view(unlabelled_indices,:) view(1,view_id).view(1,label_id).value(unlabelled_indices,:)]); 
            a_prediction_Pos(:,label_id) = a_prediction_Pos(:,label_id) .* PosterioriPos;
            a_prediction_Neg(:,label_id) = a_prediction_Neg(:,label_id) .* PosterioriNeg;
        end        
    end
    prediction_a = a_prediction_Pos >= a_prediction_Neg;
    labels(unlabelled_indices,:) = prediction_a;
    
    %UPDATE LINK VIEW            
    for link_id = 1:n_links
        tmp_view = prepare_link_view(link(link_id).links,labels);
        for label_id = 1:n_labels            
            link(link_id).view(1,label_id).value = tmp_view;             
        end
    end            
    clear tmp_view

    for link_id = 1:n_links
        for label_id = 1:n_labels
            [PosterioriPos,PosterioriNeg] = BinaryClassify_test(l_Iteration.view(link_id).classifier(label_id).model, [link(link_id).r_view(unlabelled_indices,:) link(link_id).view(label_id).value(unlabelled_indices,:)]); 
            l_prediction_Pos(:,label_id) = l_prediction_Pos(:,label_id) .* PosterioriPos;
            l_prediction_Neg(:,label_id) = l_prediction_Neg(:,label_id) .* PosterioriNeg;
        end
    end
    %prediction_r = l_prediction_Pos >= l_prediction_Neg;
    
    for label_id = 1:n_labels
        PosterioriPos = a_prediction_Pos(:,label_id) + l_prediction_Pos(:,label_id);
        PosterioriNeg = a_prediction_Neg(:,label_id) + l_prediction_Neg(:,label_id);
        prediction_1(:,label_id) = PosterioriPos >= PosterioriNeg;
    end
    
    for label_id = 1:n_labels
        PosterioriPos = a_prediction_Pos(:,label_id) .* l_prediction_Pos(:,label_id);
        PosterioriNeg = a_prediction_Neg(:,label_id) .* l_prediction_Neg(:,label_id);
        prediction_2(:,label_id) = PosterioriPos >= PosterioriNeg;
    end
    
end

