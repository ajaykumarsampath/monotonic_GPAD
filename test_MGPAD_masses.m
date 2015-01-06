%% This function generates a system with different terminal functions and constraints but
% with same size. The constraint are preconditioned accodingly.
clear all;
close all;
clc;
Nm=10; % Number of masses
T_sampling=0.5;
sys_no_precond=system_masses(Nm,struct('Ts',T_sampling,'xmin', ...
    -4*ones(2*Nm,1), 'xmax', 4*ones(2*Nm,1), 'umin', -1.5*ones(Nm,1),'umax',...
    1.5*ones(Nm,1),'b', 0.1*ones(Nm+1,1)));
various_predict_horz=10;%prediction horizon
Test_points=100;
many_points=1;
single_point=0;
x_rand=4*rand(sys_no_precond.nx,Test_points)-2;
%x_rand=[1.0184;-0.0397;-1.627;1.247;1.426;-1.211;1.987;1.293;1.705;1.609;-1.996;-0.172;1.151;1.496;-1.055;0.445;1.291;-1.829;1.884;-0.917];
result.u=zeros(sys_no_precond.nu,Test_points);
time_gpad=cell(Test_points,1);
time_mgpad=cell(Test_points,1);
time_mgpad2=cell(Test_points,1);
U_max=zeros(2,Test_points);
U_min=zeros(2,Test_points);
dual_gap=zeros(2,Test_points);
%% Generation of tree
scenario_size=[1 1 1];
for N_prb_steps=3:length(scenario_size)
    for no_of_pred=1:length(various_predict_horz)
        sys_no_precond.Np=various_predict_horz(no_of_pred);
        ops.N=sys_no_precond.Np; %step 2: argmin of the lagrangian using dynamic programming
        ops.brch_ftr=ones(ops.N,1);
        ops.brch_ftr(1:N_prb_steps)=scenario_size(1:N_prb_steps);
        Ns=prod(ops.brch_ftr);
        ops.nx=sys_no_precond.nx;
        ops.prob=cell(ops.N,1);
        for i=1:ops.N;
            if(i<=N_prb_steps)
                pd=rand(1,ops.brch_ftr(i));
                if(i==1)
                    ops.prob{i,1}=pd/sum(pd);
                    pm=1;
                else
                    pm=pm*scenario_size(i-1);
                    ops.prob{i,1}=kron(ones(pm,1),pd/sum(pd));
                end
            else
                ops.prob{i,1}=ones(1,Ns);
            end
        end
        tic
        Tree=tree_generation(ops);
        time.tree_formation=toc;
        
        SI=scenario_index(Tree);%calculation of the scenario index.
        %%
        %Cost function
        V.Q=eye(sys_no_precond.nx);
        V.R=eye(sys_no_precond.nu);
        %%terminal constraints
        sys_no_precond.Ft=cell(Ns,1);
        sys_no_precond.gt=cell(Ns,1);
        V.Vf=cell(Ns,1);
        
        r=1*rand(Ns,1);
        sys_no_precond.trm_size=(2*sys_no_precond.nx)*ones(Ns,1);
        for i=1:Ns
            %constraint in the horizon
            sys_no_precond.Ft{i}=[eye(sys_no_precond.nx);-eye(sys_no_precond.nx)];
            sys_no_precond.gt{i}=(3+0.1*rand(1))*ones(2*sys_no_precond.nx,1);
            nt=size(sys_no_precond.Ft{i},1);
            P=Polyhedron('A',sys_no_precond.Ft{i},'b',sys_no_precond.gt{i});
            if(isempty(P))
                error('Polyhedron is empty');
            end
            V.Vf{i}=dare(sys_no_precond.A,sys_no_precond.B,r(i)*V.Q,r(i)*V.R);
        end
        %% preconditioning the system and solve the system using dgpad.
        [sys,Hessian_app]=calculate_diffnt_precondition_matrix(sys_no_precond,V,Tree...
            ,struct('use_cell',1,'use_hessian',0));
        tic;
        Ptree=GPAD_dynamic_formulation_precondition(sys,V,Tree);
        toc
        ops_GPAD.steps=200;
        ops_GPAD.primal_inf=1e-3;
        ops_GPAD.dual_gap=10e-3;
        ops_GPAD.alpha=1/calculate_Lipschitz(sys,V,Tree);
        max_size=zeros(Test_points,length(Tree.stage));
        for kk=1:Test_points
            %
            ops_GPAD.x0=x_rand(:,kk);
            [Z_gpad_pre,Y_gpad_pre,time_gpad{kk}]=GPAD_differentFt_cellF(sys,Ptree,Tree,V,ops_GPAD);
            if(~isfield(time_gpad{kk},'iterate'))
                time_gpad{kk}.iterate=ops_GPAD.steps;
            end
            U_max(2,kk)=max(max(Z_gpad_pre.U));
            U_min(2,kk)=min(min(Z_gpad_pre.U));
            dual_gap(2,kk)=time_gpad{kk}.dual_gap;
            %}
            
            %monotonic behaviour
            ops_GPAD.primal_inf=1e-3;
            ops_GPAD.dual_gap=10e-3;
            %[Z_mgpad_pre,Y_mgpad_pre,time_mgpad{kk}]=MGPAD_differentFt_cellF(sys,Ptree,Tree,V,ops_GPAD);
            [Z_mgpad_pre,Y_mgpad_pre,time_mgpad{kk}]=MGPAD_differentFt_cellF...
                (sys,Ptree,Tree,V,ops_GPAD);
            if(~isfield(time_mgpad{kk},'iterate'))
                time_mgpad{kk}.iterate=ops_GPAD.steps;
            end
            U_max(1,kk)=max(max(Z_mgpad_pre.U));
            U_min(1,kk)=min(min(Z_mgpad_pre.U));
            %monotonic behaviour
            ops_GPAD.primal_inf=1e-3;
            ops_GPAD.dual_gap=10e-3;
            [Z_mgpad_pre2,Y_mgpad_pre2,time_mgpad2{kk}]=MGPAD_differentFt_cellF_var4...
                (sys,Ptree,Tree,V,ops_GPAD);
            if(~isfield(time_mgpad2{kk},'iterate'))
                time_mgpad2{kk}.iterate=ops_GPAD.steps;
            end
            U_max(3,kk)=max(max(Z_mgpad_pre2.U));
            U_min(3,kk)=min(min(Z_mgpad_pre2.U));
            dual_gap(1,kk)=time_mgpad{kk}.dual_gap;
            U_max(4,kk)=max(max(Z_mgpad_pre.U-Z_gpad_pre.U));
            U_min(4,kk)=min(max(Z_mgpad_pre.U-Z_gpad_pre.U));
            U_max(5,kk)=max(max(Z_mgpad_pre2.U-Z_gpad_pre.U));
            U_min(5,kk)=min(max(Z_mgpad_pre2.U-Z_gpad_pre.U));
        end
    end
end

%transfer_data
iterates=zeros(3,Test_points);
time_operation=zeros(3,Test_points);
dual_gap=zeros(3,Test_points);
prim_cost=zeros(3,Test_points);
for i=1:Test_points
    term_crieteria{1}(i,:)=time_mgpad{i,1}.term_crit;
    term_crieteria{2}(i,:)=time_mgpad2{i,1}.term_crit;
    term_crieteria{3}(i,:)=time_gpad{i,1}.term_crit;
    iterates(1,i)=time_gpad{i,1}.iterate;
    iterates(2,i)=time_mgpad{i,1}.iterate;
    iterates(3,i)=time_mgpad2{i,1}.iterate;
    time_operation(1,i)=time_gpad{i,1}.gpad_solve;
    time_operation(2,i)=time_mgpad{i,1}.gpad_solve;
    time_operation(3,i)=time_mgpad2{i,1}.gpad_solve;
    dual_gap(1,i)=time_gpad{i,1}.dual_gap;
    dual_gap(2,i)=time_mgpad{i,1}.dual_gap;
    dual_gap(3,i)=time_mgpad2{i,1}.dual_gap;
    prim_cost(1,i)=time_gpad{i,1}.prm_cst(end);
    prim_cost(2,i)=time_mgpad{i,1}.prm_cst(end);
    prim_cost(3,i)=time_mgpad2{i,1}.prm_cst(end);
    %{
    figure(8+i)
    subplot(211)
    plot(time_mgpad{i,1}.epsilon_prm_avg);
    hold all;
    plot(time_gpad{i,1}.epsilon_prm_avg);
    subplot(212)
    plot(time_mgpad{i,1}.epsilon_prm);
    hold all;
    plot(time_gpad{i,1}.epsilon_prm);
    %}
end
mean(time_operation')
max(time_operation')
max(iterates')
%time_operation'
%iterates'
if(single_point==1)
    for i=1:Test_points
        figure(1)
        %plot(time_gpad{i,1}.prm_cst-time_gpad{i,1}.prm_cst(1,end));
        plot(time_gpad{i,1}.prm_cst);
        title('primal cost')
        hold all;
        %plot(time_mgpad{i,1}.prm_cst-time_mgpad{i,1}.prm_cst(1,end));
        plot(time_mgpad{i,1}.prm_cst);
        plot(time_mgpad2{i,1}.prm_cst);
        legend('gpad','mono gpad','mono2 gpad')
        grid on
        figure(2)
        %plot(time_gpad{i,1}.dual_cst-time_gpad{i,1}.dual_cst(1,end));
        plot(time_gpad{i,1}.dual_cst);
        title('dual cost')
        hold all;
        %plot(time_mgpad{i,1}.dual_cst-time_mgpad{i,1}.dual_cst(1,end));
        plot(time_mgpad{i,1}.dual_cst);
        plot(time_mgpad2{i,1}.dual_cst);
        legend('gpad','mono gpad','mono2 gpad')
        grid on
        figure(3)
        plot(-(time_gpad{i,1}.prm_cst(1:end-1)-time_gpad{i,1}.prm_cst(2:end)));
        title('primal cost difference')
        hold all;
        plot(-(time_mgpad{i,1}.prm_cst(1:end-1)-time_mgpad{i,1}.prm_cst(2:end)));
        plot(-(time_mgpad2{i,1}.prm_cst(1:end-1)-time_mgpad2{i,1}.prm_cst(2:end)));
        legend('gpad','mono gpad','mono2 gpad')
        grid on
        figure(4)
        plot(-(time_gpad{i,1}.dual_cst(1:end-1)-time_gpad{i,1}.dual_cst(2:end)));
        title('dual cost difference')
        hold all;
        plot(-(time_mgpad{i,1}.dual_cst(1:end-1)-time_mgpad{i,1}.dual_cst(2:end)));
        plot(-(time_mgpad2{i,1}.dual_cst(1:end-1)-time_mgpad2{i,1}.dual_cst(2:end)));
        legend('gpad','mono gpad','mono2 gpad')
        grid on
        figure(5)
        plot(time_gpad{i,1}.dual_cst-time_gpad{i,1}.prm_cst);
        hold all;
        plot(time_mgpad{i,1}.dual_cst-time_mgpad{i,1}.prm_cst);
        plot(time_mgpad2{i,1}.dual_cst-time_mgpad2{i,1}.prm_cst);
        legend('gpad','mono gpad','mono2 gpad')
        title('dual-gap')
        grid on
        figure(6)
        plot(time_gpad{i}.epsilon_prm);
        hold all;
        grid on;
        plot(time_mgpad{i}.epsilon_prm);
        plot(time_mgpad2{i}.epsilon_prm);
        legend('gpad','mono gpad','mono2 gpad')
        title('primal infeasibility')
        figure(7)
        plot(time_gpad{i}.epsilon_prm_avg);
        hold all;
        grid on;
        plot(time_mgpad{i}.epsilon_prm_avg);
        plot(time_mgpad2{i}.epsilon_prm_avg);
        legend('gpad','mono gpad','mono2 gpad')
        title('primal average infeasibility')
        figure(8)
        plot(time_gpad{i}.dual_grad);
        hold all;
        plot(time_mgpad{i}.dual_grad);
        plot(time_mgpad2{i}.dual_grad);
        grid on;
        title('dual gradient termination condition')
        legend('gpad','mono gpad','mono2 gpad')
    end
end
%}

%%
if(many_points==1)
    figure(5)
    subplot(211)
    plot(iterates');
    legend('gpad','mono gpad','mono2 gpad')
    title('iterates')
    subplot(212)
    plot(dual_gap')
    legend('gpad','mono gpad','mono2 gpad')
    title('dual gap')
    figure(6)
    subplot(211);
    plot(U_max(1:3,:)');
    title('Infeasibility max')
    legend('gpad','mono gpad','mono2 gpad')
    subplot(212);
    plot(U_min(1:3,:)');
    title('Infeasibility min')
    legend('gpad','mono gpad','mono2 gpad')
    figure(7)
    subplot(211);
    plot(U_max(4:5,:)');
    legend('gpad-mono gpad','gpad-mono2 gpad')
    title('max contol difference')
    subplot(212)
    plot(U_min(4:5,:)');
    title('min control difference')
    figure(8)
    plot((prim_cost(1,:)-prim_cost(2,:))');
    hold all;
    plot((prim_cost(1,:)-prim_cost(3,:))');
    legend('gpad-mono gpad','gpad-mono2 gpad')
    title('primal cost difference')
end
%}