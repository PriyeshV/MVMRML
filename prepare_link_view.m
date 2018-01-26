function [label_dist] = prepare_link_view(link,labels)
    %creates relational views for links with labels from multi-view data
    [n_ids,n_labels] = size(labels);
    label_dist = zeros(n_ids,n_labels);
    
    for i = 1:n_ids
       for j = i+1:n_ids
           if link(i,j) == 1
               link(j,i) = 1;
           elseif link(j,i) == 1
               link(i,j) = 1;
            end
        end
    end
    
    for i = 1:n_ids
        neighbor = logical(link(i,:));        
        n_neighbor = nnz(neighbor);
        
        if n_neighbor == 0
            label_dist(i,:) = labels(i,:);
            if sum(label_dist(i,:)) == 0             
                label_dist(i,:) = label_dist(i,:) + 0.5;
            end
            continue;
        end
        neighbor_truth = labels(neighbor,:);
        label_dist(i,:) = sum(neighbor_truth);        
        label_dist(i,:) = label_dist(i,:) ./ n_neighbor;
        if sum(label_dist(i,:)) == 0             
                label_dist(i,:) = 1/n_labels;
        end
    end
end

