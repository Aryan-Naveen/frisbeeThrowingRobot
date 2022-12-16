recorded_waypoints = read_waypoints("backhand_recording.csv");
waypoints = transformToRobot(recorded_waypoints);
recorded_waypoints_robot = mapRobot(recorded_waypoints)

x = waypoints(:, 4:16:end)
y = waypoints(:, 8:16:end)
z = waypoints(:, 12:16:end)

x1 = recorded_waypoints_robot(:, 4:16:end);
y1 = recorded_waypoints_robot(:, 8:16:end);
z1 = recorded_waypoints_robot(:, 12:16:end);

hold on;
scatter(x, y, 'DisplayName', 'Robot Trajectory');
scatter(x1, y1, 'DisplayName', 'Tracked Trajectory');
legend();
%scatter3(recorded_waypoints(:, 8:16:end), recorded_waypoints(:, 4:16:end), recorded_waypoints(:, 12:16:end))

%set(gca, 'YDir','reverse')
xlabel("X");
ylabel("Y");
zlabel("Z");

theta(:,1) = IKshell(reshape(waypoints(1, :), 4, 4)', [1.5291 -0.4803    0.4122   -3.1779   -1.4916    1.5533]);
%theta(:,1) = IKshell([0 -1 0 waypoints(1, 4); 0 0 1 waypoints(1, 8); -1 0 0 waypoints(1, 12); 0 0 0 1], [1.5291 -0.4803    0.4122   -3.1779   -1.4916    1.5533]);
for i=2:size(waypoints, 1)
    fprintf("Computing inverse kinematics for waypoint %d\n", i);
    theta(:,i) = IKshell(reshape(waypoints(i, :), 4, 4)', theta(:,i-1));
    %theta(:,i) = IKshell([0 -1 0 waypoints(i, 4); 0 0 1 waypoints(1, 8); -1 0 0 waypoints(i, 12); 0 0 0 1], theta(:, i-1));
end

theta

%{
velCmd = rospublisher("/joint_group_vel_controller/command");
velMsg = rosmessage(velCmd);
jSub = rossubscriber('joint_states');

dt = 0.25;

for i=1:size(theta, 2)-1
    jMsg = receive(jSub);
    angles = jMsg.Position([3 2 1 4 5 6]);
    velMsg.Data = ((theta(:,i+1)-angles) / dt);axes

    send(velCmd, velMsg);
    pause(dt);
end

velMsg.Data = zeros(6, 1);
send(velCmd, velMsg);
%}

function waypoints = read_waypoints(file)
    M = readtable(file);
    M = removevars(M, 'Var1');

    waypoints = zeros(size(M));

    for i = 1:size(M, 1)
        waypoints(i, :) = table2array(M(i, :));
        waypoints(i, 4) = waypoints(i, 4) * 1000;
        waypoints(i, 8) = waypoints(i, 8) * 1000;
        waypoints(i, 12) = waypoints(i, 12) * 1000;
    end
end

function mapToRobotCoordinates = mapRobot(waypoints)
    T0r = [0 1 0 -632;
               0 0 1 720;
               1 0 0 439;
               0 0 0 1];
    Trc = [0,     0,    1,   0;
           1,     0,    0,    0;
           0,     -1,   0,    0;
           0,     0,    0,    1];

    mapToRobotCoordinates = zeros(size(waypoints));

    for i = 1:size(waypoints, 1)
        Ttc = reshape(waypoints(i, :), 4, 4)';
%        Ttr = T0r * Trc * Tci * inv(Trc * inv(Ttc));
        Ttr = inv(Trc * inv(Ttc));
        robotT = Ttr;
        mapToRobotCoordinates(i, :) = reshape(robotT', 1, 16);
    end
end

function transformed = transformToRobot(waypoints)
    T0r = [0 1 0 -632;
               0 0 1 720;
               1 0 0 439;
               0 0 0 1];
    Trc = [0,     0,    1,   0;
           1,     0,    0,    0;
           0,     -1,   0,    0;
           0,     0,    0,    1];

    transformed = zeros(size(waypoints));
    Ttci = reshape(waypoints(1, :), 4, 4)';

    for i = 1:size(waypoints, 1)
        Ttc = reshape(waypoints(i, :), 4, 4)';
%        Ttr = T0r * Trc * Tci * inv(Trc * inv(Ttc));
        T = T0r * Trc * inv(Ttci);
        Ttr = T * Ttc * inv(Trc);
        robotT = Ttr;
        transformed(i, :) = reshape(robotT', 1, 16);
    end
end