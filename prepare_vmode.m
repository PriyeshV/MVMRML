function [ l_corr ] = prepare_vmode(labels,label_id )
    %returns the label correlations
    [n_ids,n_labels] = size(labels);
    o_labels = setdiff(1:n_labels,label_id);
    %o_labels = 1:n_labels;
    l_corr = labels(:,o_labels);
end

