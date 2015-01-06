function [ Z,Y,details] = MGPAD_differentFt_cellF_var4(sys,Ptree,Tree,V,ops)
%This function is implements the GPAD algorithm to solve an optimization
%problem. The inputs the system are the system dynamics, the tree
%structure, all the off-line elements that are calculated earlier that are
%to be used by the GPAD. additional options about the inital fesible dual
%variable.

%The terminal constraints are different.

% In this algorithm the dual variables correspond to nodes of the tree.

% Monotinicity by checking the gradient of the dual space.

Ns=length(Tree.leaves);% total scenarios in the tree
Nd=length(Tree.stage);%toal nodes in the tree
% Initalizing the dual varibables
Y.y0=zeros(Nd-Ns,size(sys.F{1},1));
Y.y1=zeros(Nd-Ns,size(sys.F{1},1));
prm_fes_term=cell(Ns,1);
epsilon_prm=1;
for i=1:Ns
    Y.yt0{i,:}=zeros(1,size(sys.Ft{i,1},1));
    Y.yt1{i,:}=zeros(1,size(sys.Ft{i,1},1));
    prm_fes_term{i,1}=zeros(size(sys.Ft{i,1},1),1);
end


prm_fes=zeros(size(sys.F{1},1),Nd-Ns);
g_nodes=zeros(size(sys.F{1},1),Nd-Ns);
for i=1:Nd-Ns
    g_nodes(:,i)=sys.g{i,1};
end
g_nodes_term=sys.gt;
theta=[1 1]';
tic
j=1;
W_minyt=zeros(Ns,1);
details.restart=0;
details.term_crit=zeros(1,4);
while(j<ops.steps)
    % Step 1: accelerated step
    W.y=Y.y1+theta(2)*(1/theta(1)-1)*(Y.y1-Y.y0);
    
    for i=1:Ns
        W.yt{i,1}=Y.yt1{i,1}+theta(2)*(1/theta(1)-1)*(Y.yt1{i,1}-Y.yt0{i,1});
        W_minyt(i,1)=min(W.yt{i,1});
    end
    
    
    %step 2: argmin of the lagrangian using dynamic programming
    [Z,Q]=GPAD_dynamic_calculation_Ft(sys,Ptree,Tree,W,ops.x0);
    
    
    %step 3: Projection of y on the positive quadrant.
    Y.y0=Y.y1;
    Y.yt0=Y.yt1;
    for i=1:Tree.ancestor(Tree.leaves(end))
        prm_fes(:,i)=sys.F{i}*Z.X(:,i)+sys.G{i}*Z.U(:,i);
        Y.y1(i,:)=(max(0,W.y(i,:)'+ops.alpha*(prm_fes(:,i)-g_nodes(:,i))))';
    end
    for i=1:length(Tree.leaves)
        prm_fes_term{i,1}=sys.Ft{i,1}*Z.X(:,Tree.leaves(i));
        Y.yt1{i,1}=max(0,W.yt{i,:}+ops.alpha*(prm_fes_term{i,1}-sys.gt{i,1})');
    end
    
    %check for restart
    iter=j;
    details.dual_grad(iter)=0;
    for i=1:Nd-Ns
        details.dual_grad(iter)=details.dual_grad(iter)+(prm_fes(:,i)-g_nodes(:,i))'*(Y.y1(i,:)-Y.y0(i,:))';
    end
    for i=1:Ns
        details.dual_grad(iter)=details.dual_grad(iter)+(Y.yt1{i,1}-Y.yt0{i,1})*...
            (prm_fes_term{i,1}-g_nodes_term{i,1});
    end
    if(j>1)
        if(mod(details.restart,2)==0)
            if(details.dual_grad(iter)<0)
                details.restart=details.restart+1;
                details.restart_iter(details.restart)=iter;
                %theta_t=theta;
                theta=[1 1]';
                W.y=Y.y1+theta(2)*(1/theta(1)-1)*(Y.y1-Y.y0);
                %Wt.y=Y.y1;
                for i=1:Ns
                    W.yt{i,1}=Y.yt1{i,1}+theta(2)*(1/theta(1)-1)*(Y.yt1{i,1}-Y.yt0{i,1});
                    W_minyt(i,1)=min(W.yt{i,1});
                    %Wt.yt{i,1}=Y.yt1{i,1};
                end
                Z=GPAD_dynamic_calculation_Ft(sys,Ptree,Tree,W,ops.x0);
                %step 3: Projection of y on the positive quadrant.
                Y.y0=Y.y1;
                Y.yt0=Y.yt1;
                for i=1:Tree.ancestor(Tree.leaves(end))
                    prm_fes(:,i)=sys.F{i}*Z.X(:,i)+sys.G{i}*Z.U(:,i);
                    Y.y1(i,:)=(max(0,W.y(i,:)'+ops.alpha*(prm_fes(:,i)-g_nodes(:,i))))';
                end
                for i=1:length(Tree.leaves)
                    prm_fes_term{i,1}=sys.Ft{i,1}*Z.X(:,Tree.leaves(i));
                    Y.yt1{i,1}=max(0,W.yt{i,:}+ops.alpha*(prm_fes_term{i,1}-sys.gt{i,1})');
                end
                
                details.dual_grad(iter)=0;
                for i=1:Nd-Ns
                    details.dual_grad(iter)=details.dual_grad(iter)+(prm_fes(:,i)-g_nodes(:,i))'*(Y.y1(i,:)-Y.y0(i,:))';
                end
                for i=1:Ns
                    details.dual_grad(iter)=details.dual_grad(iter)+(Y.yt1{i,1}-Y.yt0{i,1})*...
                        (prm_fes_term{i,1}-g_nodes_term{i,1});
                end
                %theta=theta_t;
            end
        else
            if(details.dual_grad(iter)>0)
                details.restart=details.restart+1;
                details.restart_iter(details.restart)=iter;
                %theta_t=theta;
                theta=[1 1]';
                W.y=Y.y1+theta(2)*(1/theta(1)-1)*(Y.y1-Y.y0);
                %Wt.y=Y.y1;
                for i=1:Ns
                    W.yt{i,1}=Y.yt1{i,1}+theta(2)*(1/theta(1)-1)*(Y.yt1{i,1}-Y.yt0{i,1});
                    W_minyt(i,1)=min(W.yt{i,1});
                    %Wt.yt{i,1}=Y.yt1{i,1};
                end
                Z=GPAD_dynamic_calculation_Ft(sys,Ptree,Tree,W,ops.x0);
                %step 3: Projection of y on the positive quadrant.
                Y.y0=Y.y1;
                Y.yt0=Y.yt1;
                for i=1:Tree.ancestor(Tree.leaves(end))
                    prm_fes(:,i)=sys.F{i}*Z.X(:,i)+sys.G{i}*Z.U(:,i);
                    Y.y1(i,:)=(max(0,W.y(i,:)'+ops.alpha*(prm_fes(:,i)-g_nodes(:,i))))';
                end
                for i=1:length(Tree.leaves)
                    prm_fes_term{i,1}=sys.Ft{i,1}*Z.X(:,Tree.leaves(i));
                    Y.yt1{i,1}=max(0,W.yt{i,:}+ops.alpha*(prm_fes_term{i,1}-sys.gt{i,1})');
                end
                
                details.dual_grad(iter)=0;
                for i=1:Nd-Ns
                    details.dual_grad(iter)=details.dual_grad(iter)+(prm_fes(:,i)-g_nodes(:,i))'*(Y.y1(i,:)-Y.y0(i,:))';
                end
                for i=1:Ns
                    details.dual_grad(iter)=details.dual_grad(iter)+(Y.yt1{i,1}-Y.yt0{i,1})*...
                        (prm_fes_term{i,1}-g_nodes_term{i,1});
                end
                %theta=theta_t;
            end
        end
    end
    
    %termination criteria
    if(j==1)
        prm_avg_next=prm_fes;
        prm_avg_term_next=prm_fes_term;
        epsilon_prm_avg=max( max(max(prm_fes-g_nodes)), ...
            max(max(cell2mat(prm_fes_term)-cell2mat(g_nodes_term))) );
    else
        prm_avg_next=(1-theta(2))*prm_avg_next+theta(2)*prm_fes;
        for m=1:Ns
            prm_avg_term_next{m,1}=(1-theta(2))*prm_avg_term_next{m,1}+theta(2)*prm_fes_term{m,1};
        end
        epsilon_prm_avg=max(max(max(prm_avg_next-g_nodes)),...
            max(max(cell2mat(prm_avg_term_next)-cell2mat(g_nodes_term))));
    end
    if (epsilon_prm_avg<=ops.primal_inf) %average_primal feasibility less&& details.restart<1
        if(details.restart<1)
            details.term_crit(1,1)=1;
            details.iterate=j;
            j=10*ops.steps;
            %
        else
            if (min(min(min(W.y)),min( W_minyt))>0)
                sum=0;
                for i=1:Nd-Ns
                    sum=sum-W.y(i,:)*(prm_fes(:,i)-g_nodes(:,i));
                end
                for i=1:Ns
                    sum=sum-W.yt{i,:}*(prm_fes_term{i,1}-g_nodes_term{i,1});
                end
                if sum<=ops.dual_gap %condition 29. dual gap
                    details.term_crit(1,2)=1;
                    details.iterate=j;
                    j=10*ops.steps;
                else
                    prm_cst=0;%primal cost;
                    for i=1:Nd-Ns
                        prm_cst=prm_cst+Tree.prob(i,1)*(Z.X(:,i)'*V.Q*Z.X(:,i)+Z.U(:,i)'*V.R*Z.U(:,i));
                    end
                    for i=1:Ns
                        prm_cst=prm_cst+Tree.prob(Tree.leaves(i))*(Z.X(:,Tree.leaves(i))'*V.Vf{i,1}*...
                            Z.X(:,Tree.leaves(i)));
                    end
                    if sum<=ops.dual_gap*prm_cst/(1+ops.dual_gap) %condition 30 dual gap
                        details.term_crit(1,3)=1;
                        details.iterate=j;
                        j=10*ops.steps;
                    else
                        %step 4: theta update
                        theta(1)=theta(2);
                        theta(2)=(sqrt(theta(1)^4+4*theta(1)^2)-theta(1)^2)/2;
                        j=j+1;
                    end
                end
            else
                prm_cst=0;%primal cost;
                dual_cst=0;% dual cost;
                for i=1:Nd-Ns
                    prm_cst=prm_cst+Tree.prob(i,1)*(Z.X(:,i)'*V.Q*Z.X(:,i)+Z.U(:,i)'*V.R*Z.U(:,i));
                    dual_cst=dual_cst+Y.y1(i,:)*(prm_fes(:,i)-g_nodes(:,i));
                end
                for i=1:Ns
                    prm_cst=prm_cst+Tree.prob(Tree.leaves(i))*(Z.X(:,Tree.leaves(i))'*V.Vf{i,1}*...
                        Z.X(:,Tree.leaves(i)));
                    dual_cst=dual_cst+Y.yt1{i,:}*(prm_fes_term{i,1}-g_nodes_term{i,1});
                end
                if (-dual_cst<=ops.dual_gap*max(dual_cst,1)) %condtion 27 (dual gap)
                    details.term_crit(1,4)=1;
                    details.iterate=j;
                    j=10*ops.steps;
                else
                    %step 4: theta update
                    theta(1)=theta(2);
                    theta(2)=(sqrt(theta(1)^4+4*theta(1)^2)-theta(1)^2)/2;
                    j=j+1;
                end
            end
        end
        %}
    else
        epsilon_prm=max( max(max(prm_fes-g_nodes)), ...
            max(max(cell2mat(prm_fes_term)-cell2mat(g_nodes_term))) );
        if(epsilon_prm<=ops.primal_inf) % primal feasibility of the iterate
            if (min(min(min(W.y)),min( W_minyt))>0)
                sum=0;
                for i=1:Nd-Ns
                    sum=sum-W.y(i,:)*(prm_fes(:,i)-g_nodes(:,i));
                end
                for i=1:Ns
                    sum=sum-W.yt{i,:}*(prm_fes_term{i,1}-g_nodes_term{i,1});
                end
                if sum<=ops.dual_gap %condition 29. dual gap
                    details.term_crit(1,2)=1;
                    details.iterate=j;
                    j=10*ops.steps;
                else
                    prm_cst=0;%primal cost;
                    for i=1:Nd-Ns
                        prm_cst=prm_cst+Tree.prob(i,1)*(Z.X(:,i)'*V.Q*Z.X(:,i)+Z.U(:,i)'*V.R*Z.U(:,i));
                    end
                    for i=1:Ns
                        prm_cst=prm_cst+Tree.prob(Tree.leaves(i))*(Z.X(:,Tree.leaves(i))'*V.Vf{i,1}*...
                            Z.X(:,Tree.leaves(i)));
                    end
                    if sum<=ops.dual_gap*prm_cst/(1+ops.dual_gap) %condition 30 dual gap
                        details.term_crit(1,3)=1;
                        details.iterate=j;
                        j=10*ops.steps;
                    else
                        %step 4: theta update
                        theta(1)=theta(2);
                        theta(2)=(sqrt(theta(1)^4+4*theta(1)^2)-theta(1)^2)/2;
                        j=j+1;
                    end
                end
            else
                prm_cst=0;%primal cost;
                dual_cst=0;% dual cost;
                for i=1:Nd-Ns
                    prm_cst=prm_cst+Tree.prob(i,1)*(Z.X(:,i)'*V.Q*Z.X(:,i)+Z.U(:,i)'*V.R*Z.U(:,i));
                    dual_cst=dual_cst+Y.y1(i,:)*(prm_fes(:,i)-g_nodes(:,i));
                end
                for i=1:Ns
                    prm_cst=prm_cst+Tree.prob(Tree.leaves(i))*(Z.X(:,Tree.leaves(i))'*V.Vf{i,1}*...
                        Z.X(:,Tree.leaves(i)));
                    dual_cst=dual_cst+Y.yt1{i,:}*(prm_fes_term{i,1}-g_nodes_term{i,1});
                end
                if (-dual_cst<=ops.dual_gap*max(dual_cst,1)) %condtion 27 (dual gap)
                    details.term_crit(1,4)=1;
                    details.iterate=j;
                    j=10*ops.steps;
                else
                    %step 4: theta update
                    theta(1)=theta(2);
                    theta(2)=(sqrt(theta(1)^4+4*theta(1)^2)-theta(1)^2)/2;
                    j=j+1;
                end
            end
        else
            %step 4: theta update
            theta(1)=theta(2);
            theta(2)=(sqrt(theta(1)^4+4*theta(1)^2)-theta(1)^2)/2;
            j=j+1;
        end
    end
    
    details.prm_cst(iter)=0;%primal cost 
    details.dual_cst(iter)=0;%dual cost  
    for i=1:Nd-Ns
        details.dual_cst(iter)=details.dual_cst(iter)+Y.y1(i,:)*(prm_fes(:,i)-g_nodes(:,i));
        details.prm_cst(iter)=details.prm_cst(iter)+Tree.prob(i,1)*(Z.X(:,i)'*V.Q*Z.X(:,i)+Z.U(:,i)'*V.R*Z.U(:,i));
    end
    for i=1:Ns
        details.prm_cst(iter)=details.prm_cst(iter)+Tree.prob(Tree.leaves(i))*(Z.X(:,Tree.leaves(i))'*V.Vf{i,1}*...
            Z.X(:,Tree.leaves(i)));
        details.dual_cst(iter)=details.dual_cst(iter)+Y.yt1{i,1}*(prm_fes_term{i,1}-g_nodes_term{i,1});
    end
    details.dual_cst(iter)=details.prm_cst(iter)-details.dual_cst(iter);
    
    details.epsilon_prm_avg(iter)=epsilon_prm_avg;
    details.epsilon_prm(iter)=epsilon_prm;
end
details.dual_gap=details.prm_cst(iter)-details.dual_cst(iter);
details.gpad_solve=toc;
details.W=W;
%details.Q=Q;
%details.epsilon_prm_avg= epsilon_prm_avg;
%details.epsilon_prm=epsilon_prm;
end












