clear all;
%Choose dataset
datasets = {'Twitter_Data/'};%Movie_Data/'};

%Dataset format
%name_views.mat - Attributes views in n_samples*n_features format
%name_links.mat - Relational views in adjacency matrix format

if strcmp(datasets,'Twitter_Data/')
    name_views = {'tweets','lists','listmerged'};
    name_links = {'follows','followedby','mentions','mentionedby','retweets','retweetedby'};
elseif strcmp(datasets,'Movie_Data/')
    name_views = {'Tags','Summary'};
    name_links = {'Actor_Graph','Director_Graph'};
end

%Paramaeters
train_percentage = [0.1 0.3 0.5 0.7 0.9];
v_mode = 2;
l_mode = 1;

distfun='euclidean';

Learner.type = 'SVM';
Learner.para = 'none';
Learner.attri_type = 1;

%Number of folds
K = 5;
%--------------------------------------------------------------------------
for d = 1:size(datasets,2)    
    str = sprintf('Running experiment for %s',datasets{d});
    disp(str);
    %load ids
    load(char(strcat(datasets(d),'raw_ids.mat')));
    n_ids = size(ids,1);
    
    %load truth labels
    truth = load(char(strcat(datasets(d),'truth.mat')));
    truth = truth.('truth');
    n_labels = size(truth,2);
    truth(truth == -1) = 0;
 
    for train_perc = train_percentage
        %Save results to *_Data/ folder under results_** name
        % * - Dataset name
        % ** - labeled data ratio
        fid = fopen(char(strcat(datasets(d),'results_',num2str(train_perc*100),'.txt')),'w');
        str = sprintf('\t %d percentage of Labeled data ',train_perc*100);
        disp(str)

        %Calculate Perofmance measures
        accuracy = zeros(K,1);
		ex_accuracy = zeros(K,1);
        h_accuracy = zeros(K,1);
        precision = zeros(K,1);
        recall = zeros(K,1);
        f_measure = zeros(K,1);
		label_acc = zeros(K,1);

        %cross-validate
        for k = 1:K           
            %disp(strcat('K ============',num2str(k)));
            str = sprintf('\t\t %d fold',k);
            disp(str)
            load(char(strcat(datasets(d),'labelled_indices_perc_',num2str(train_perc*100),'/',num2str(k),'.mat')));
    
            labelled_indices = logical(labelled_indices);
            unlabelled_indices  = ~labelled_indices;
            n_labelled = nnz(labelled_indices);
            n_unlabelled  = nnz(unlabelled_indices);

            V_Iteration = struct;
            V_Iteration(1,1).labelled_indices = false(n_ids,n_labels);    
            V_Iteration(1,1).unlabelled_indices = false(n_ids,n_labels);   
            V_Iteration(1,1).view = struct;
            V_Iteration(2,1).view = struct;
            V_Iteration(1,1).view.classifier = struct;
            V_Iteration(2,1).view.classifier = struct;

            %prepare views
            n_views = length(name_views);     
            view = struct;  
            v_iter_id = 1;               
         
            for view_id = 1:n_views        
                tmp_obj = load(char(strcat(datasets(d),name_views(1,view_id),'.mat')));        
                tmp = tmp_obj.('view');
                view(1,view_id).r_view = tmp;
                %view(1,view_id).r_view = tmp ./ repmat(max(tmp),n_ids,1);

                for label_id = 1:n_labels            
                    view(1,view_id).view(1,label_id).value = zeros(n_ids,0);
                    if v_mode == 2 % + label_correlation
                        view(1,view_id).view(1,label_id).value = zeros(n_ids,n_labels-1);
                    end
                end

            end       

            L_Iteration = struct;
            L_Iteration(1,1).labelled_indices = false(n_ids,n_labels);
            L_Iteration(1,1).unlabelled_indices = false(n_ids,n_labels);
            L_Iteration(1,1).view = struct;
            L_Iteration(2,1).view = struct;
            L_Iteration(1,1).view.classifier = struct;
            L_Iteration(2,1).view.classifier = struct;

            %prepare links
            n_links = length(name_links);			
            link = struct;

            for link_id = 1:n_links              
                tmp_obj = load(char(strcat(datasets(d),name_links(1,link_id),'.mat')));
                link(link_id).links = tmp_obj.('links');     
                link(link_id).r_view = zeros(n_ids,0);

                for label_id = 1:n_labels            
                    link(link_id).view(1,label_id).value = zeros(n_ids,n_labels); % label_distributions                                
                end
            end
            clear tmp_obj; 
            %----------------------------------------------------------------------
                    
            for label_id = 1:n_labels
                    V_Iteration(1,1).labelled_indices(:,label_id) = labelled_indices;    
                    V_Iteration(1,1).unlabelled_indices(:,label_id) = unlabelled_indices;   
            end    

            %Initialization
                for view_id = 1:n_views  
                    V_Iteration(1,v_iter_id).view(1,view_id).classifier = struct;
                    V_Iteration(1,v_iter_id).view(1,view_id).predicted_labels = zeros(n_ids,n_labels);                
                    V_Iteration(1,v_iter_id).view(1,view_id).predicted_labels(labelled_indices,:) = truth(labelled_indices,:);            

                    for label_id = 1:n_labels
      
                        %Training
                        V_Iteration(1,v_iter_id).view(1,view_id).classifier(label_id).model = BinaryClassify_train(Learner,[view(1,view_id).r_view(labelled_indices,:) view(1,view_id).view(1,label_id).value(labelled_indices,:)],V_Iteration(1,v_iter_id).view(1,view_id).predicted_labels(labelled_indices,label_id));            

                        [PosterioriPos,PosterioriNeg] = BinaryClassify_test(V_Iteration(1,v_iter_id).view(1,view_id).classifier(label_id).model,[view(1,view_id).r_view(unlabelled_indices,:)  view(1,view_id).view(1,label_id).value(unlabelled_indices,:)]);
                        V_Iteration(1,v_iter_id).view(1,view_id).predicted_labels(unlabelled_indices,label_id) = PosterioriPos>=PosterioriNeg;                             
                        V_Iteration(1,v_iter_id).view(1,view_id).classifier(label_id).PosterioriPos = PosterioriPos; %unlabelled - posterior probabilities               
                    end
                end
                %Append Label correlations
                for view_id = 1:n_views
                    for label_id = 1:n_labels 
                        label_correlations = prepare_vmode(V_Iteration(1,v_iter_id).view(1,view_id).predicted_labels,label_id);
                        label_correlations(label_correlations == 0) = -1;
                        label_correlations = 1 - label_correlations;
                        if v_mode == 2
                         	   view(1,view_id).view(1,label_id).value(:,end-(n_labels-2):end) = label_correlations;
                        end
                    end
                end
                v_iter_id = 2;
                %disp('MV_ML Learning');
                V_Iteration(1,v_iter_id) = CoTraining_STEP(Learner, view, [V_Iteration(1,v_iter_id-1) V_Iteration(1,1)], 1);   
            
                %------------------------------------------------------------------

                %disp('Starting Multi-view Multi-Relational Multi-Label Learning');
                l_iter_id = 1;
                view_prediction = zeros(n_ids,n_labels);        
                link_prediction = zeros(n_ids,n_labels);

                L_Iteration(1,1).labelled_indices = V_Iteration(1,v_iter_id).labelled_indices;    
                L_Iteration(1,1).unlabelled_indices = V_Iteration(1,v_iter_id).unlabelled_indices;                        
                %UPDATE LINK VIEW     
                tmp = get_prediction_views(V_Iteration(1,v_iter_id));  
                for label_id = 1:n_labels
                    view_prediction(V_Iteration(1,v_iter_id).labelled_indices(:,label_id),label_id) = V_Iteration(1,v_iter_id).view(1).predicted_labels( V_Iteration(1,v_iter_id).labelled_indices(:,label_id),label_id);
                    view_prediction(V_Iteration(1,v_iter_id).unlabelled_indices(:,label_id),label_id) = tmp(:,label_id);             
                end        
                for link_id = 1:n_links
                    tmp_view = prepare_link_view(link(link_id).links,view_prediction);
                    for label_id = 1:n_labels
                        if l_mode == 1
                            link(link_id).view(1,label_id).value = tmp_view;
                        end                   
                    end
                end            
                clear tmp_view

                for link_id = 1:n_links 
                    L_Iteration(1,l_iter_id).view(1,link_id).classifier = struct;
                    L_Iteration(1,l_iter_id).view(1,link_id).predicted_labels = zeros(n_ids,n_labels);                

                    for label_id = 1:n_labels           

                        L_Iteration(1,l_iter_id).view(1,link_id).predicted_labels(L_Iteration(1,l_iter_id).labelled_indices(:,label_id),label_id) = view_prediction( L_Iteration(1,l_iter_id).labelled_indices(:,label_id),label_id);                                     
                        L_Iteration(1,l_iter_id).view(1,link_id).classifier(label_id).model = BinaryClassify_train(Learner, link(link_id).view(1,label_id).value(labelled_indices,:), L_Iteration(1,l_iter_id).view(1,link_id).predicted_labels(labelled_indices,label_id));            

                        [PosterioriPos,PosterioriNeg] = BinaryClassify_test(L_Iteration(1,l_iter_id).view(1,link_id).classifier(label_id).model, link(link_id).view(1,label_id).value(L_Iteration(1,l_iter_id).unlabelled_indices(:,label_id),:));
                        L_Iteration(1,l_iter_id).view(1,link_id).predicted_labels(L_Iteration(1,1).unlabelled_indices(:,label_id),label_id) = PosterioriPos>=PosterioriNeg;
                        L_Iteration(1,l_iter_id).view(1,link_id).classifier(label_id).PosterioriPos = PosterioriPos; %unlabelled - posterior probabilities
                    end
                end

                %Start Semi-supervised learning
                while ( 1 == 1)
                   V_Iteration(2,v_iter_id) = V_Iteration(1,1);           
                   V_Iteration(2,v_iter_id).labelled_indices = L_Iteration(1,l_iter_id).labelled_indices;
                   V_Iteration(2,v_iter_id).unlabelled_indices = L_Iteration(1,l_iter_id).unlabelled_indices; 
                   %UPDATE ATTRIBUTE VIEW'S LABEL CORRELATIONS 
                   for label_id = 1:n_labels
                       for view_id = 1:n_views
                            V_Iteration(2,v_iter_id).view(1,view_id).predicted_labels = V_Iteration(1,v_iter_id).view(1,view_id).predicted_labels;
                            V_Iteration(2,v_iter_id).view(1,view_id).predicted_labels(V_Iteration(2,v_iter_id).labelled_indices(:,label_id),label_id) = L_Iteration(1,l_iter_id).view(1).predicted_labels(V_Iteration(2,v_iter_id).labelled_indices(:,label_id),label_id);
                            
                            label_correlations = prepare_vmode(V_Iteration(1,v_iter_id).view(1,view_id).predicted_labels,label_id);
                            label_correlations(label_correlations == 0) = -1;
                            label_correlations = 1 - label_correlations;
                            if v_mode == 2
                                   view(1,view_id).view(1,label_id).value(:,end-(n_labels-2):end) = label_correlations;
                            end
                       end
                   end                     
                   
                  % disp('Co-regularization MV_ML Learner with the confidence of MR_ML Learner');
                   V_Iteration(2,v_iter_id) = CoTraining_STEP(Learner, view, [L_Iteration(1,l_iter_id) V_Iteration(2,v_iter_id)], 1);
                   %===============================================================
                   if sum(V_Iteration(2,l_iter_id).unlabelled_indices) == 0
                       terminate = 2;
                       break;
                   end
                   %UPDATE ATTRIBUTE VIEW'S LABEL CORRELATIONS 
                   for label_id = 1:n_labels
                       for view_id = 1:n_views
                            label_correlations = prepare_vmode(V_Iteration(1,v_iter_id).view(1,view_id).predicted_labels,label_id);
                            label_correlations(label_correlations == 0) = -1;
                            label_correlations = 1 - label_correlations;
                            if v_mode == 2
                                   view(1,view_id).view(1,label_id).value(:,end-(n_labels-2):end) = label_correlations;
                            end
                       end
                   end
                   %===============================================================
                   v_iter_id = v_iter_id + 1;
                   %disp('MV_ML Learning');
                   V_Iteration(1,v_iter_id) = CoTraining_STEP(Learner, view, [V_Iteration(2,v_iter_id-1) V_Iteration(1,1)], 1);       
                   %===============================================================
                   if sum(V_Iteration(1,l_iter_id).unlabelled_indices) == 0
                       terminate = 1;
                       break;
                   end
                   
                   l_iter_id = l_iter_id + 1;
                   L_Iteration(1,l_iter_id) = L_Iteration(1,1);           
                   L_Iteration(1,l_iter_id).labelled_indices = V_Iteration(1,v_iter_id).labelled_indices;
                   L_Iteration(1,l_iter_id).unlabelled_indices = V_Iteration(1,v_iter_id).unlabelled_indices; 
                   %link views obtains information from MV_ML
                   %UPDATE LABELS                    
                   for label_id = 1:n_labels
                       for link_id = 1:n_links
                            L_Iteration(1,l_iter_id).view(1,link_id).predicted_labels = L_Iteration(1,l_iter_id).view(1,link_id).predicted_labels;
                            L_Iteration(1,l_iter_id).view(1,link_id).predicted_labels(L_Iteration(1,l_iter_id).labelled_indices(:,label_id),label_id) = V_Iteration(1,v_iter_id).view(1).predicted_labels(L_Iteration(1,l_iter_id).labelled_indices(:,label_id),label_id);
                       end
                   end
                    %UPDATE LINK VIEW           
                    tmp = get_prediction_views(V_Iteration(1,v_iter_id));    
                    for label_id = 1:n_labels
                        view_prediction(L_Iteration(1,l_iter_id).labelled_indices(:,label_id),label_id) = V_Iteration(1,l_iter_id).view(1).predicted_labels(L_Iteration(1,l_iter_id).labelled_indices(:,label_id),label_id);
                        view_prediction(L_Iteration(1,l_iter_id).unlabelled_indices(:,label_id),label_id) = tmp(:,label_id);
                    end
                    for link_id = 1:n_links
                        tmp_view = prepare_link_view(link(link_id).links,view_prediction);
                        for label_id = 1:n_labels                    
                            if l_mode == 1
                                link(link_id).view(1,label_id).value = tmp_view;
                            end                   
                        end
                    end            
                    clear tmp_view
                    
                    %disp('MR_ML learning);
                    for link_id = 1:n_links 
                        for label_id = 1:n_labels 
                            L_Iteration(1,l_iter_id).view(1,link_id).classifier(label_id).model = BinaryClassify_train(Learner, link(link_id).view(1,label_id).value(L_Iteration(1,l_iter_id).labelled_indices(:,label_id),:), L_Iteration(1,l_iter_id).view(1,link_id).predicted_labels(L_Iteration(1,l_iter_id).labelled_indices(:,label_id),label_id));            

                            [PosterioriPos,PosterioriNeg] = BinaryClassify_test(L_Iteration(1,l_iter_id).view(1,link_id).classifier(label_id).model, link(link_id).view(1,label_id).value(L_Iteration(1,l_iter_id).unlabelled_indices(:,label_id),:));
                            L_Iteration(1,l_iter_id).view(1,link_id).predicted_labels(L_Iteration(1,l_iter_id).unlabelled_indices(:,label_id),label_id) = PosterioriPos>=PosterioriNeg;
                            L_Iteration(1,l_iter_id).view(1,link_id).classifier(label_id).PosterioriPos = PosterioriPos; %unlabelled - posterior probabilities
                        end
                    end
                end     
               % disp('Semi-supervised learning by CoTraining Done!!!');
                [~,prediction_2,~] = get_Predictions_CoTraining(unlabelled_indices, view, link, V_Iteration(terminate(1),v_iter_id), L_Iteration(1,l_iter_id),V_Iteration(1,1),L_Iteration(1,1),2,truth);
                [accuracy(k,1),recall(k,1),precision(k,1),f_measure(k,1),h_accuracy(k,1),ex_accuracy(k,1),label_acc(k,1)] = calc_acc_CoTraining(truth(unlabelled_indices,:),prediction_2);             
                fprintf(fid,'%f %f %f %f %f %f %f\n',[accuracy(k,1),precision(k,1),recall(k,1),f_measure(k,1),h_accuracy(k,1),ex_accuracy(k,1),label_acc(k,1)]);
                
           end
          disp([accuracy precision recall f_measure h_accuracy ex_accuracy label_acc]);
          fprintf(fid,'%f %f %f %f %f %f %f\n',[mean(accuracy) mean(precision) mean(recall) mean(f_measure) mean(h_accuracy) mean(ex_accuracy) mean(label_acc)]);
          %disp('save done');
          fclose(fid);
    end
end 
