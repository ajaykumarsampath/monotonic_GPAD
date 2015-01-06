%% This function generates a system with different terminal functions and constraints but
% with same size. The constraint are preconditioned accodingly.
clear all;
close all;
clc;
Nm=20; % Number of masses
T_sampling=0.25;
sys_no_precond=system_generation(Nm,struct('Ts',T_sampling,'xmin', ...
    -5*ones(2*Nm,1), 'xmax', 5*ones(2*Nm,1), 'umin', -1*ones(Nm,1),'umax',...
    1*ones(Nm,1)));
various_predict_horz=10;%prediction horizon
Test_points=100;
x_rand=2*(1.8*rand(sys_no_precond.nx,Test_points)-0.9);
result.u=zeros(sys_no_precond.nu,Test_points);
time_gpad=cell(Test_points,1);
time_gpd=cell(Test_points,1);
time_mgpad=cell(Test_points,1);
U_max=zeros(3,Test_points);
U_min=zeros(3,Test_points);
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
        for i=1:ops.N
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
        ops_GPAD.steps=500;
        ops_GPAD.primal_inf=1e-3;
        ops_GPAD.dual_gap=10e-4;
        ops_GPAD.alpha=1/calculate_Lipschitz(sys,V,Tree);
        max_size=zeros(Test_points,length(Tree.stage));
        for kk=1:Test_points
            ops_GPAD.x0=x_rand(:,kk);
            [Z_gpad_pre,Y_gpad_pre,time_gpad{kk}]=GPAD_differentFt_cellF(sys,Ptree,Tree,V,ops_GPAD);
            if(~isfield(time_gpad{kk},'iterate'))
                time_gpad{kk}.iterate=ops_GPAD.steps;
            end
            U_max(1,kk)=max(max(Z_gpad_pre.U));
            U_min(1,kk)=min(min(Z_gpad_pre.U)); 
            %without accelearation step
            [Z_gpd_pre,Y_gpd_pre,time_gpd{kk}]=GPD_differentFt_cellF(sys,Ptree,Tree,V,ops_GPAD);
            if(~isfield(time_gpd{kk},'iterate'))
                time_gpd{kk}.iterate=ops_GPAD.steps;
            end
            U_max(2,kk)=max(max(Z_gpd_pre.U));
            U_min(2,kk)=min(min(Z_gpd_pre.U)); 
            %monotonic behaviour
            ops_GPAD.primal_inf=1e-3;
            ops_GPAD.dual_gap=1e-4;
            [Z_mgpad_pre,Y_mgpad_pre,time_mgpad{kk}]=MGPAD_differentFt_cellF(sys,Ptree,Tree,V,ops_GPAD);
            if(~isfield(time_mgpad{kk},'iterate'))
                time_mgpad{kk}.iterate=ops_GPAD.steps;
            end
            U_max(3,kk)=max(max(Z_mgpad_pre.U));
            U_min(3,kk)=min(min(Z_mgpad_pre.U)); 
        end
    end
end

%transfer_data
iterates=zeros(3,Test_points);
time_operation=zeros(3,Test_points);
for i=1:Test_points
    iterates(1,i)=time_gpad{i,1}.iterate;
    iterates(2,i)=time_gpd{i,1}.iterate;
    iterates(3,i)=time_mgpad{i,1}.iterate;
    time_operation(1,i)=time_gpad{i,1}.gpad_solve;
    time_operation(2,i)=time_gpd{i,1}.gpad_solve;
    time_operation(3,i)=time_mgpad{i,1}.gpad_solve;
end
mean(time_operation')
%{
for i=1:Test_points
    figure(1)
    plot(time_gpad{i,1}.prm_cst-time_gpad{i,1}.prm_cst(1,end));
    title('primal cost')
    hold all;
    plot(time_gpd{i,1}.prm_cst-time_gpd{i,1}.prm_cst(1,end));
    plot(time_mgpad{i,1}.prm_cst-time_mgpad{i,1}.prm_cst(1,end));
    figure(2)
    plot(time_gpad{i,1}.dual_cst-time_gpad{i,1}.dual_cst(1,end));
    title('dual cost')
    hold all;
    plot(time_gpd{i,1}.dual_cst-time_gpd{i,1}.dual_cst(1,end));
    plot(time_mgpad{i,1}.dual_cst-time_mgpad{i,1}.dual_cst(1,end));
    figure(3)
    plot(-(time_gpad{i,1}.prm_cst(1:end-1)-time_gpad{i,1}.prm_cst(2:end)));
    title('primal cost difference')
    hold all;
    plot(-(time_gpd{i,1}.prm_cst(1:end-1)-time_gpd{i,1}.prm_cst(2:end)));
    plot(-(time_mgpad{i,1}.prm_cst(1:end-1)-time_mgpad{i,1}.prm_cst(2:end)));
    figure(4)
    plot(-(time_gpad{i,1}.dual_cst(1:end-1)-time_gpad{i,1}.dual_cst(2:end)));
    title('dual cost difference')
    hold all;
    plot(-(time_gpd{i,1}.dual_cst(1:end-1)-time_gpd{i,1}.dual_cst(2:end)));
    plot(-(time_mgpad{i,1}.dual_cst(1:end-1)-time_mgpad{i,1}.dual_cst(2:end)));
end
%}

figure(5)
plot(iterates(1,:));
hold all;
plot(iterates(2,:));
plot(iterates(3,:));
title('Iterates')
legend('GPAD','DUAL-Projection','MGPAD');
figure(6)
subplot(211);
plot(U_max(1,:));
hold all;
plot(U_max(2,:));
plot(U_max(3,:));
legend('GPAD','DUAL-Projection','MGPAD');
subplot(212);
plot(U_min(1,:));
hold all;
plot(U_min(2,:));
plot(U_min(3,:));
title('Infeasibility')
legend('GPAD','DUAL-Projection','MGPAD');
%}