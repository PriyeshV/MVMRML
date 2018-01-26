function [sequence] = select_topK_2(n_ids,ratio,Pos, Neg )
    %k 'll be the 10% of the labeled data
    n_ul = size(Pos,1);  
    n_l = n_ids - n_ul;
    k = ceil(0.10*n_l);
    if n_ul < k
        k = n_ul;
    end
    
    n_pos = ceil(ratio * k);
    n_neg = abs(k - n_pos);
    
        
    [~,ind] = sort(Pos,'descend');
    pos_sequence = ind(1:n_pos,1);
    
    while 1 == 1
        [~, ind] = sort(Neg,'descend');
        neg_sequence = ind(1:n_neg,1);
        sequence = [pos_sequence; neg_sequence];
        if size(unique(sequence),1) == k
            break;
        end
        n_neg = n_neg + 1;
    end
end

