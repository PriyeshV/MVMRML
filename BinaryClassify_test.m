function [PosterioriPos,PosterioriNeg]=BinaryClassify_test(Classifier,data)
%BinaryClassify_test classifiers data points into binary classes
%
%    Syntax
%
%       [PosterioriPos,PosterioriNeg]=BinaryClassify_test(Classifier,data)
%
%    Description
%
%       BinaryClassify_test takes,
%           Classifier      - A struct variable with three fields 'type', 'attri_type' and 'model' that specifies the classifier's relevant information:
%                             1) Classifier.type gives the type of classifiers, which can take the value of 'NB' (naive bayes), 'BP' (bp neural
%                                network) or 'CART' (CART decision tree);
%                             2) Classifier.attri_type indicates which kinds of attributes it can deal with, 0 for binary features while 1 for
%                                real-valued features;
%                             3) Classifier.model contains the specific model information:
%                                a) If strcmp(Classifier.type,'NB')==1, Classifier.model is again a struct variable with three fields 'prior', 'paraPos' (2xd1 array)
%                                   and 'paraNeg' (2xd1 array):
%                                   a1) Classifier.model.prior(1) gives the prior probability of an instance being positive while Classifier.model.prior(2)
%                                       gives the prior probability of an instance being negative.
%                                   a2) The meanings of the other two fields (i.e. ParaPos and paraNeg) depend on the value of Classifier.attri_type 
%                                       i)  If Classifier.attri_type is 0 (binary features), Classifier.model.paraPos(1,d) gives the conditional probability that an instance
%                                           will take a value of 1 on its d-th dimension given it is positive, while Classifier.model.paraPos(2,d) gives the conditional
%                                           probability that an instance will take a value of 0 on its d-th dimension given it is positive. Similarly, Classifier.model.paraNeg(1,d)
%                                           gives the conditional probability that an instance will take a value of 1 on its d-th dimension given it is negative, while 
%                                           Classifier.model.paraNeg(2,d) gives the conditional probability that an instance will take a value of 0 on its d-th
%                                           dimension given it is negative.
%                                       ii) If Classifier.attri_type is 1 (real-valued features), Classifier.model.paraPos(1,d) gives the mean of the Gaussian distribution on the 
%                                           instance's d-th dimension given it is positive, while Classifier.model.paraPos(2,d) gives the standard deviation of the
%                                           Gaussian distribution on the instance's d-th dimension given it is positive. Similarly, Classifier.model.paraNeg(1,d) 
%                                           gives the mean of the Gaussian distribution on the instance's d-th dimension given it is negative, while Classifier.model.paraNeg(2,d) 
%                                           gives the standard deviation of the Gaussian distribution on the instance's d-th dimension given it is negative.
%                                b) If strcmp(Classifier.type,'SVM')==1, Classifier.model is a struct variable used by Libsvm [1]
%                                c) If strcmp(Classifier.type,'CART')==1, Classifier.model is a tree struct variable used by MATLAB (see Statistics Toolbox
%                                   on classification and regression trees)
%
%           data        	- An Mxd array, where the i-th data to be classified is stored in data(i,:)
%
%      and returns,
%           PosterioriPos   - An Mx1 vector, where PosterioriPos(i,1) gives the posterior probability that the i-th instance being positive
%           PosterioriNeg   - An Mx1 vector, where PosterioriNeg(i,1) gives the posterior probability that the i-th instance being negative. It is assumed that
%                             PosterioriPos(i,1)+PosterioriNeg(i,1)=1
%
% [1] C.-C. Chang and C.-J. Lin. LIBSVM: A library for support vector machines, 2001. software available at http://www.csie.ntu.edu.tw/~cjlin/libsvm

    num_inst=size(data,1);
    
    switch Classifier.type
        case 'NB'
            Log_Prior_Pos=log(Classifier.model.prior(1)*ones(num_inst,1));
            Log_Prior_Neg=log(Classifier.model.prior(2)*ones(num_inst,1));
            if(Classifier.attri_type==0)
                Log_Likeli_Pos=sum(log(data.*concur(Classifier.model.paraPos(1,:)',num_inst)'+(1-data).*concur(Classifier.model.paraPos(2,:)',num_inst)'),2);
                Log_Likeli_Neg=sum(log(data.*concur(Classifier.model.paraNeg(1,:)',num_inst)'+(1-data).*concur(Classifier.model.paraNeg(2,:)',num_inst)'),2);
            else
                Log_Likeli_Pos=sum(log(normpdf(data,concur(Classifier.model.paraPos(1,:)',num_inst)',concur(Classifier.model.paraPos(2,:)',num_inst)')),2);
                Log_Likeli_Neg=sum(log(normpdf(data,concur(Classifier.model.paraNeg(1,:)',num_inst)',concur(Classifier.model.paraNeg(2,:)',num_inst)')),2);
            end
            PosterioriPos=1./(1+exp(Log_Prior_Neg+Log_Likeli_Neg-Log_Prior_Pos-Log_Likeli_Pos));
            PosterioriPos=min(PosterioriPos,ones(num_inst,1));
        case 'CART'
            [YFit,Nodes]=eval(Classifier.model,data);
            Prob=classprob(Classifier.model,Nodes);
            if(size(Prob,2)==1)
                if(YFit{1,1}==0)
                    PosterioriPos=zeros(num_inst,1);
                else
                    PosterioriPos=ones(num_inst,1);
                end
            else
                PosterioriPos=Prob(:,2);
            end
        case 'SVM'
            [predicted_label,accuracy,prob_estimates]=libsvmpredict(ones(num_inst,1),data,Classifier.model,'-b 1');
            pos_index=find(Classifier.model.Label);
            PosterioriPos=prob_estimates(:,pos_index);
        otherwise
            PosterioriNeg = posterior(Classifier.model,data);
            PosterioriPos = PosterioriNeg(:,2);
            PosterioriNeg = PosterioriNeg(:,1);
            return;
    end
    
    PosterioriNeg=1-PosterioriPos;