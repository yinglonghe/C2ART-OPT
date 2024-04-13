close all
clear
clc

% [33333, 1458, 8422, 50010]
carIDs = int64([33333, 33333, 1458, 8422, 50010, 1458, 8422, 50010]);   % Including the platoon leader
numVeh = length(carIDs);                    % The number of vehicles in a platoon (incl. the leader)
numFoll = numVeh-1;
timeStep = 0.1;                          % Time step
timeSim = 10;                            % Trip duration
numStep = floor(timeSim/timeStep);  % Simulation setps
Np = 20;                                 % Number of prediction horizon (steps)
d  = 20;                                 % Desired spacing
Time = (0.1:timeStep:timeSim);

[a_max, a_min, f_0, f_1, f_2, phi, tau_a, veh_mass, mfc_array, mfc_slice] = parameter_init(carIDs(2:end));

% ['PF', 'PLF', 'BD', 'BDL', 'TPF', 'TPLF']
[matA, matP] = communication_init(numFoll, 'TPLF');

[Position, Velocity, Acceleration, U, p0, v0, a0, Pa, Va, Aa, ua,... 
    Pa_next, Va_next, Aa_next, ua_next, Pend, Vend, Aend, Cost, Exitflg]...
    = variable_init(numFoll, numStep, timeStep, Np, d, f_0, f_1, f_2,...
        phi, tau_a, veh_mass, mfc_array, mfc_slice); 

for i = 2:numStep-Np
    fprintf('\n Steps i= %d\n',i)
    for j = 1:numFoll
        tic
        sin_theta = 0;
        
        [vehType, mfc_curve, X0, R, Xa, F, numXdes, Xdes, Q, numXn, Xn, G, lb, ub, u0, Pnp, Vnp, Anp]...
            = mpc_step_init(i, j, Np, d, numFoll, Position, Velocity, Acceleration,...
            a_max, a_min, f_0, f_1, f_2, phi, tau_a, veh_mass, mfc_array, mfc_slice,...
            Pa, Va, Aa, ua, matA, matP, p0, v0, a0);
        
        Pend(j, i) = Pnp;
        Vend(j, i) = Vnp;
        Aend(j, i) = Anp;
        
        tol_opt = 1e-5;
        options = optimset('Display','off', 'TolFun', tol_opt, 'MaxIter', 2000,...
            'LargeScale', 'off', 'RelLineSrchBnd', [], 'RelLineSrchBndDuration', 1);
        A = [];b = []; Aeq = []; beq = [];
        [u, Cost(j, i), Exitflg(j, i), output] = fmincon(...
            @(u) cost_func(u, Np, timeStep, sin_theta, X0, vehType, mfc_curve, Q, Xdes, R, F, Xa, G, Xn, numXn),...
            u0, A, b, Aeq, beq, lb, ub, @(u) nonlcon_func(u, Np, timeStep, sin_theta, X0, vehType, mfc_curve, Pnp, Vnp, Anp),options); 
        
        [U, Position, Velocity, Acceleration, Pa_next, Va_next, Aa_next, ua] = assume_update(i, j, Np, u, U, ua, Position, Velocity, Acceleration,...
            timeStep, sin_theta, vehType, mfc_curve, Pa_next, Va_next, Aa_next);
        
        toc
    end
    
    Pa = Pa_next;
    Va = Va_next;
    Aa = Aa_next;

end

ResPlot(numFoll, numStep, Np, d, U, Position, Velocity, Acceleration, p0, v0, a0)


function ResPlot(numFoll, numStep, Np, d, U, Position, Velocity, Acceleration, p0, v0, a0)

figure;
plot(p0(1:numStep-Np) - Position(1, 1:numStep-Np), 'DisplayName', 'Spacing_1');hold on;
for k = 2:numFoll
    plot(Position(k-1, 1:numStep-Np) - Position(k, 1:numStep-Np), 'DisplayName', ['Spacing_' num2str(k)]);hold on;
end
legend
xlabel('Step');
ylabel('Spacing (m)');

figure;
for k = 1:numFoll
    plot(Position(k, 1:numStep-Np) - (p0(1:numStep-Np) - k*d), 'DisplayName', ['ErrSpacLead_' num2str(k)]);hold on;
end
legend
xlabel('Step');
ylabel('Lead spacing error (m)');

figure;
plot(Position(1, 1:numStep-Np) - (p0(1:numStep-Np) - d), 'DisplayName', 'ErrSpacNbr_1');hold on;
for k = 2:numFoll
    plot(Position(k, 1:numStep-Np) - (Position(k-1, 1:numStep-Np) - d), 'DisplayName', ['ErrSpacNbr_' num2str(k)]);hold on;
end
legend
xlabel('Step');
ylabel('Neighbor spacing error (m)');

figure;
plot(v0(1:numStep-Np), 'DisplayName', 'v_0');hold on;
for k = 1:numFoll
    plot(Velocity(k, 1:numStep-Np), 'DisplayName', ['v_' num2str(k)]);hold on;
end
legend
xlabel('Step');
ylabel('Speed (m/s)');

figure;
plot(a0(1:numStep-Np), 'DisplayName', 'a_0');hold on;
for k = 1:numFoll
    plot(Acceleration(k, 1:numStep-Np), 'DisplayName', ['a_' num2str(k)]);hold on;
end
legend
xlabel('Step');
ylabel('Acceleration (m/s^2)');

figure;
for k = 1:numFoll
    plot(U(k, 1:numStep-Np), 'DisplayName', ['U_' num2str(k)]);hold on;
end
legend
xlabel('Step');
ylabel('U - acceleration (m/s^2)');

end


function [U, Position, Velocity, Acceleration, Pa_next, Va_next, Aa_next, ua] = assume_update(i, j, Np, u, U, ua, Position, Velocity, Acceleration,...
    timeStep, sin_theta, vehType, mfc_curve, Pa_next, Va_next, Aa_next)

f_0 = vehType(1);
f_1 = vehType(2);
f_2 = vehType(3);
phi = vehType(4);
tau_a = vehType(5);
veh_mass = vehType(6);
mfc_curve = mat2npndarray(mfc_curve);

U(j, i) = u(1);

RES = py.c2art_env.sim_env.platooning_mpc.setup_utils.vehicle_dynamic(...
    U(j, i),...
    Position(j, i-1),...
    Velocity(j, i-1),...
    Acceleration(j, i-1),...
    timeStep,...
    sin_theta,...
    f_0,...
    f_1,...
    f_2,...
    phi,...
    tau_a,...
    veh_mass,...
    mfc_curve);

Position(j, i) = RES{1};
Velocity(j, i) = RES{2};
Acceleration(j, i) = RES{3};

Temp = zeros(3, Np+1);
Temp(:, 1) = [Position(j, i); Velocity(j, i); Acceleration(j, i)];   
ua(j, 1:Np-1) = u(2:Np);
for k = 1:Np-1
    RES = py.c2art_env.sim_env.platooning_mpc.setup_utils.vehicle_dynamic(...
        ua(j, k),...
        Temp(1, k),...
        Temp(2, k),...
        Temp(3, k),...
        timeStep,...
        sin_theta,...
        f_0,...
        f_1,...
        f_2,...
        phi,...
        tau_a,...
        veh_mass,...
        mfc_curve);
    Temp(1, k+1) = RES{1};
    Temp(2, k+1) = RES{2};
    Temp(3, k+1) = RES{3};
end

ua(j, Np) = (f_0 * (1 - sin_theta^2)^0.5 + f_1 * Temp(2, Np) + f_2 * Temp(2, Np)^2) / veh_mass + 9.80665 * sin_theta;

RES = py.c2art_env.sim_env.platooning_mpc.setup_utils.vehicle_dynamic(...
    ua(j, Np),...
    Temp(1, Np),...
    Temp(2, Np),...
    Temp(3, Np),...
    timeStep,...
    sin_theta,...
    f_0,...
    f_1,...
    f_2,...
    phi,...
    tau_a,...
    veh_mass,...
    mfc_curve);

Temp(1, Np+1) = RES{1};
Temp(2, Np+1) = RES{2};
Temp(3, Np+1) = RES{3};

Pa_next(j,:) = Temp(1,:);
Va_next(j,:) = Temp(2,:);
Aa_next(j,:) = Temp(3,:);

end


function [C, Ceq] = nonlcon_func(u, Np, timeStep, sin_theta, X0, vehType, mfc_curve, Pnp, Vnp, Anp)

u = mat2np1darray(u);
Np = int16(Np);
% timeStep
% sin_theta
X0 = mat2np1darray(X0);
vehType = mat2np1darray(vehType);
mfc_curve = mat2npndarray(mfc_curve);
% Pnp
% Vnp
% Anp

RES = py.c2art_env.sim_env.platooning_mpc.setup_utils.nonlcon_func(...
    u,...
    Np,...
    timeStep,...
    sin_theta,...
    X0,...
    vehType,...
    mfc_curve,...
    Pnp,...
    Vnp,...
    Anp);

C = [];
Ceq = nparray2mat(RES);

end


function cost = cost_func(u, Np, timeStep, sin_theta, X0, vehType, mfc_curve, Q, Xdes, R, F, Xa, G, Xn, numXn)

u = mat2np1darray(u);
Np = int16(Np);
% timeStep
% sin_theta
X0 = mat2np1darray(X0);
vehType = mat2np1darray(vehType);
mfc_curve = mat2npndarray(mfc_curve);
Q = mat2npndarray(Q);
Xdes = mat2npndarray(Xdes);
% R
F = mat2npndarray(F);
Xa = mat2npndarray(Xa);
G = mat2npndarray(G);
Xn = mat2npndarray(Xn);
numXn = int16(numXn);

RES = py.c2art_env.sim_env.platooning_mpc.setup_utils.cost_func(...
    u,...
    Np,...
    timeStep,...
    sin_theta,...
    X0,...
    vehType,...
    mfc_curve,...
    Q,...
    Xdes,...
    R,...
    F,...
    Xa,...
    G,...
    Xn,...
    numXn);

cost = RES;

end


function [vehType, mfc_curve, X0, R, Xa, F, numXdes, Xdes, Q, numXn, Xn, G, lb, ub, u0, Pnp, Vnp, Anp] = mpc_step_init(...
    i, j, Np, d, numFoll, Position, Velocity, Acceleration, a_max, a_min, f_0, f_1, f_2, ...
    phi, tau_a, veh_mass, mfc_array, mfc_slice, Pa, Va, Aa, ua, matA, matP, p0, v0, a0)

vehType = [f_0(j), f_1(j), f_2(j), phi(j), tau_a(j), veh_mass(j)];
mfc_curve = mfc_array(:, mfc_slice(j)+1:mfc_slice(j+1));

X0 = [Position(j, i-1), Velocity(j, i-1), Acceleration(j, i-1)];

% Input-deviation
R = 0.1;

% Self-deviation
Xa = [Pa(j, :); Va(j, :); Aa(j, :)];
F = diag([5.0, 2.5, 1.0]) .* (sum(matA(:, j))+1)^2;

% Leader-deviation
if matP(j, j) == 1
    numXdes = 1;
    Pd = p0(i-1:i+Np-1) - j*d;
    Vd = v0(i-1:i+Np-1);
    Ad = a0(i-1:i+Np-1);
    Xdes = [Pd; Vd; Ad];
    Q = diag([5.0, 2.5, 1.0]);
else
    numXdes = 0;
    Xdes = zeros(3, Np+1);
    Q = diag([0.0, 0.0, 0.0]);
end

% Neighbor-deviation
numXn = sum(matA(j, :));
if numXn == 0
    Xn = zeros(3, Np+1);
    G = diag([0.0, 0.0, 0.0]);
else
    Xn = zeros(0, Np+1);
    for k = 1:numFoll
        if matA(j,k) == 1
            Xn = [Xn; Pa(k,:) - (j-k)*d; Va(k,:); Aa(k,:)];    
        end
    G = diag([5.0, 2.5, 1.0]);
    end
end

lb = a_min(j) * ones(1,Np);
ub = a_max(j) * ones(1,Np);
u0 = ua(j,:);

Pnp = Xdes(1, end);
Vnp = Xdes(2, end);
Anp = Xdes(3, end);

for k = 1:numXn
    Pnp = Pnp + Xn(3*k-2, end);
    Vnp = Vnp + Xn(3*k-1, end);
    Anp = Anp + Xn(3*k, end);
end
Pnp = Pnp/(numXdes+numXn);
Vnp = Vnp/(numXdes+numXn);
Anp = Anp/(numXdes+numXn);
       
end


function [Position, Velocity, Acceleration, U, p0, v0, a0, Pa, Va, Aa, ua, Pa_next, Va_next, Aa_next, ua_next, Pend, Vend, Aend, Cost, Exitflg] = variable_init(numFoll, numStep, timeStep, Np, d, f_0, f_1, f_2, phi, tau_a, veh_mass, mfc_array, mfc_slice)  
 
numFoll = int16(numFoll);
numStep = int16(numStep);
% timeStep
Np = int16(Np);
% d
f_0 = mat2np1darray(f_0);
f_1 = mat2np1darray(f_1); 
f_2 = mat2np1darray(f_2);
phi = mat2np1darray(phi);
tau_a = mat2np1darray(tau_a);
veh_mass = mat2np1darray(veh_mass);
mfc_array = mat2npndarray(mfc_array);
mfc_slice = mat2np1darray(mfc_slice);

RES = py.c2art_env.sim_env.platooning_mpc.setup_utils.variable_init(...
    numFoll, numStep, timeStep, Np, d, f_0, f_1, f_2,...
    phi, tau_a, veh_mass, mfc_array, mfc_slice);

Position = nparray2mat(RES{1});
Velocity = nparray2mat(RES{2});
Acceleration = nparray2mat(RES{3});
U = nparray2mat(RES{4});
p0 = nparray2mat(RES{5});
v0 = nparray2mat(RES{6});
a0 = nparray2mat(RES{7});
Pa = nparray2mat(RES{8});
Va = nparray2mat(RES{9});
Aa = nparray2mat(RES{10});
ua = nparray2mat(RES{11});
Pa_next = nparray2mat(RES{12});
Va_next = nparray2mat(RES{13});
Aa_next = nparray2mat(RES{14});
ua_next = nparray2mat(RES{15});
Pend = nparray2mat(RES{16});
Vend = nparray2mat(RES{17});
Aend = nparray2mat(RES{18});
Cost = nparray2mat(RES{19});
Exitflg = nparray2mat(RES{20});

end
 
 
function [matA, matP] = communication_init(numFoll, TopoType)

numFoll = int16(numFoll);

% ['PF', 'PLF', 'BD', 'BDL', 'TPF', 'TPLF']
RES = py.c2art_env.sim_env.platooning_mpc.setup_utils.communication_init(numFoll, TopoType);
matA = nparray2mat(RES{1});
matP = nparray2mat(RES{2});

end


function [a_max, a_min, f_0, f_1, f_2, phi, tau_a, veh_mass, mfc_array, mfc_slice] = parameter_init(follIDs)

% ['PF', 'PLF', 'BD', 'BDL', 'TPF', 'TPLF']
RES = py.c2art_env.sim_env.platooning_mpc.setup_utils.parameter_init(follIDs);

a_max = nparray2mat(RES{1}); 
a_min = nparray2mat(RES{2});
f_0 = nparray2mat(RES{3});
f_1 = nparray2mat(RES{4});
f_2 = nparray2mat(RES{5});
phi = nparray2mat(RES{6});
tau_a = nparray2mat(RES{7});
veh_mass = nparray2mat(RES{8});
mfc_array = nparray2mat(RES{9});
mfc_slice = int16(nparray2mat(RES{10}));

end


function result = mat2np1darray( matarray )

result=py.numpy.array(matarray);

end


function result = mat2npndarray( matarray )
  %mat2nparray Convert a Matlab array into an nparray
  %   Convert an n-dimensional Matlab array into an equivalent nparray  
  data_size=size(matarray);
  if length(data_size)==1
      % 1-D vectors are trivial
      result=py.numpy.array(matarray);
  elseif length(data_size)==2
      % A transpose operation is required either in Matlab, or in Python due
      % to the difference between row major and column major ordering
      transpose=matarray';
      % Pass the array to Python as a vector, and then reshape to the correct
      % size
      result=py.numpy.reshape(transpose(:)', int32(data_size));
  else
      % For an n-dimensional array, transpose the first two dimensions to
      % sort the storage ordering issue
      transpose=permute(matarray,[length(data_size):-1:1]);
      % Pass it to python, and then reshape to the python style of matrix
      % sizing
      result=py.numpy.reshape(transpose(:)', int32(fliplr(size(transpose))));
  end
end


function result = nparray2mat( nparray )
  %nparray2mat Convert an nparray from numpy to a Matlab array
  %   Convert an n-dimensional nparray into an equivalent Matlab array
  data_size = cellfun(@int64,cell(nparray.shape));
  if length(data_size)==1
      % This is a simple operation
      result=double(py.array.array('d', py.numpy.nditer(nparray)));
  elseif length(data_size)==2
      % order='F' is used to get data in column-major order (as in Fortran
      % 'F' and Matlab)
      result=reshape(double(py.array.array('d', ...
          py.numpy.nditer(nparray, pyargs('order', 'F')))), ...
          data_size);
  else
      % For multidimensional arrays more manipulation is required
      % First recover in python order (C contiguous order)
      result=double(py.array.array('d', ...
          py.numpy.nditer(nparray, pyargs('order', 'C'))));
      % Switch the order of the dimensions (as Python views this in the
      % opposite order to Matlab) and reshape to the corresponding C-like
      % array
      result=reshape(result,fliplr(data_size));
      % Now transpose rows and columns of the 2D sub-arrays to arrive at the
      % correct Matlab structuring
      result=permute(result,[length(data_size):-1:1]);
  end
end