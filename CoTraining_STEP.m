function [Iteration] = CoTraining_STEP(Learner,view,Iteration,mode)
    
    n_views = size(view,2);
    [n_ids,n_labels] = size(Iteration(1).view(1).predicted_labels);
    
    labelled_indices = Iteration(1).labelled_indices;
    unlabelled_indices = Iteration(1).unlabelled_indices;
    n_unlabelled = nnz(unlabelled_indices(:,1));
    tmp_PosterioriPos = struct;
    tmp_PosterioriNeg = struct;
    
    labels = false(n_unlabelled,n_labels);
    v_labels = zeros(n_ids,n_labels);
    
    for label_id = 1:n_labels                
            tmp_PosterioriPos(label_id).value = zeros(n_unlabelled,1);
            tmp_PosterioriNeg(label_id).value = zeros(n_unlabelled,1);            
           
            n_labelled = (n_ids - n_unlabelled);
            pos_ratio = nnz(Iteration(1).view(1).predicted_labels(labelled_indices(:,label_id),label_id))/n_labelled;
            
            if mode == 1 %(Finds predictions to be updated to the training set)
                %combine unlabelled instances' prediction from other views
                id = 1;                
                for o_view_id = 1:size(Iteration(1).view(1),2)            
                    tmp_PosterioriPos(label_id).value = tmp_PosterioriPos(label_id).value + Iteration(id).view(o_view_id).classifier(label_id).PosterioriPos;
                    tmp_PosterioriNeg(label_id).value = tmp_PosterioriNeg(label_id).value + (1 - Iteration(id).view(o_view_id).classifier(label_id).PosterioriPos);                              
                end                
                tmp_PosterioriPos(label_id).value = tmp_PosterioriPos(label_id).value/n_views;
                tmp_PosterioriNeg(label_id).value = tmp_PosterioriNeg(label_id).value/n_views;
                
                labels(:,label_id) = tmp_PosterioriPos(label_id).value >= tmp_PosterioriNeg(label_id).value;
                
                %Find top-k points to updated to the training set
                update_indices = select_topK_2(n_ids,pos_ratio,tmp_PosterioriPos(label_id).value, tmp_PosterioriNeg(label_id).value);
                unlabelled = find([1:n_ids]'.*unlabelled_indices(:,label_id));
                updates = unlabelled(update_indices);
                
                labelled_indices(updates,label_id) = 1;
                unlabelled_indices(:,label_id) = ~labelled_indices(:,label_id); 
                
                Iteration(2).labelled_indices(:,label_id) = labelled_indices(:,label_id);
                Iteration(2).unlabelled_indices(:,label_id) = unlabelled_indices(:,label_id);
        
                v_labels(:,label_id) = Iteration(1).view(1).predicted_labels(:,label_id);
                v_labels(updates,label_id) = labels(update_indices,label_id);
          
            elseif mode == 2 %(Just uses labels from the other view to update the training set)
                v_labels(:,label_id) = Iteration(2).view(1).predicted_labels(:,label_id);
                
                labelled_indices(:,label_id) = Iteration(2).labelled_indices(:,label_id);
                unlabelled_indices(:,label_id) = Iteration(2).unlabelled_indices(:,label_id);                 
             end
    end       
    
    for label_id = 1:n_labels   
        for view_id = 1:n_views         
           Iteration(2).view(view_id).classifier(label_id).model = BinaryClassify_train(Learner, [view(view_id).r_view(labelled_indices(:,label_id),:) view(view_id).view(label_id).value(labelled_indices(:,label_id),:)], v_labels(labelled_indices(:,label_id),label_id) );
           [PosterioriPos,PosterioriNeg] = BinaryClassify_test(Iteration(2).view(view_id).classifier(label_id).model, [view(view_id).r_view(unlabelled_indices(:,label_id),:) view(view_id).view(label_id).value(unlabelled_indices(:,label_id),:)]);
           Iteration(2).view(view_id).predicted_labels(unlabelled_indices(:,label_id),label_id) = PosterioriPos>=PosterioriNeg;
           Iteration(2).view(view_id).classifier(label_id).PosterioriPos = PosterioriPos;
        end 
    end 
    Iteration = Iteration(2);
end