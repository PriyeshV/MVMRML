function [acc,recall,precision,f_measure,h_accuracy,ex_acc,label_acc] = calc_acc_CoTraining(truth,prediction)
     
     [n_unlabelled,n_labels] = size(truth);
     hamming_dist = zeros(n_unlabelled,1);
     
     ex_acc = 0;
     acc = 0;
     recall = 0;
     precision = 0;
     f_measure = 0;
     label_acc = 0;
     
     n = n_unlabelled;
     for i = 1:n_unlabelled
          same_labels = 0;
           hamming_dist(i,1) = pdist([prediction(i,:);truth(i,:)],'hamming');         
           t_labels = find(truth(i,:));           
           n_t_l = size(t_labels,2); 
           
           p_labels = find(prediction(i,:));
           n_p_l = size(p_labels,2);
           
           ex_acc = ex_acc + (sum(prediction(i,:) == truth(i,:)) == n_labels);
                      
            if n_t_l == 0 && isempty(find(prediction(i,:), 1))
                n = n - 1;
                continue;
            end
            if n_p_l == 0
                continue;
            end
            for p_l = p_labels
                if sum(p_l == find(truth(i,:))) ~= 0
                    same_labels = same_labels + 1;
                end
            end
            acc = acc + (same_labels / size(union(t_labels,p_labels),2));
            recall = recall + (same_labels/n_t_l);
            precision = precision + (same_labels/n_p_l);
            f_measure = f_measure + ((2*same_labels)/(n_t_l + n_p_l));
     end
     
     for i = 1:n_labels
         label_acc = label_acc + mean(prediction(:,i) == truth(:,i));
     end
     label_acc = label_acc/n_labels;
     
     h_accuracy = 1 - mean(hamming_dist);
     ex_acc = ex_acc/n;
     acc = acc/n;
     recall = recall/n;
     precision = precision/n;
     f_measure = f_measure/n;     
end

