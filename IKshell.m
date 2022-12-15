function theta = IKshell(T,theta0)
    % Numerical calculation of UR5e inverse kinematics for end-effector position described by the transformation matrix T, starting from initial guess theta0 for the angles.
    
    % First just make sure theta0 is a column vector, and exit if not
    if size(theta0,1)==1, theta0 = theta0'; elseif size(theta0,2)~=1, disp('Initial guess needs to be a 1D vector'); return, end
    
    % Repeating the arm geometry from the FK lab; all values in mm:
    W2 = 259.6;
    W1 = 133.3;
    H2 = 99.7;
    H1 = 162.5;
    L1 = 425;
    L2 = 392.2;
    
    % Screw axes in the world frame (note that these are different from those given in the book in Example 4.5, but that's because the book uses different directions for both the world and body frames than we did in the FK lab):
    w1 = [0 0 1];
    w2 = [0 1 0];
    w3 = [0 1 0];
    w4 = [0 1 0];
    w5 = [0 0 -1];
    w6 = [0 1 0];
    q1 = [0 0 0];
    q2 = [0 0 H1];
    q3 = [L1 0 H1];
    q4 = [L1+L2 0 H1];
    q5 = [L1+L2 W1 H1];
    q6 = [L1+L2 0 H1-H2];
    v1 = cross(-w1,q1);
    v2 = cross(-w2,q2);
    v3 = cross(-w3,q3);
    v4 = cross(-w4,q4);
    v5 = cross(-w5,q5);
    v6 = cross(-w6,q6);
    S1 = [w1 v1];
    S2 = [w2 v2];
    S3 = [w3 v3];
    S4 = [w4 v4];
    S5 = [w5 v5];
    S6 = [w6 v6];
    %S1 = [0, 0, 1, 0, 0, 0]';
    %S2 = [0, 1, 0, -H1, 0, 0]';
    %S3 = [0, 1, 0, -H1, 0, L1]';
    %S4 = [0, 1, 0, -H1, 0, L1+L2]';
    %S5 = [0, 0, -1, -W1, L1+L2, 0]';
    %S6 = [0, 1, 0, H2-H1, 0, L1+L2]';
    M = [-1 0 0 (L1+L2); 0 0 1 (W1+W2); 0 1 0 H1-H2; 0 0 0 1];
    % Transformation between the world and body frames (when at home position) -- M above and T below should be (and are) the same:
    Rab = [ [-1 0 0]' [0 0 1]' [0 1 0]' ];
    Tab = [Rab [(L1+L2) (W1+W2) H1-H2]'; 0 0 0 1];
    % Bracketed versions, S1 -> [S1] etc (definition at Eqn 3.85 in the book):
    S1b = [0 -S1(3) S1(2) S1(4) ; S1(3) 0 -S1(1) S1(5) ; -S1(2) S1(1) 0 S1(6) ; 0 0 0 0];
    S2b = [0 -S2(3) S2(2) S2(4) ; S2(3) 0 -S2(1) S2(5) ; -S2(2) S2(1) 0 S2(6) ; 0 0 0 0];
    S3b = [0 -S3(3) S3(2) S3(4) ; S3(3) 0 -S3(1) S3(5) ; -S3(2) S3(1) 0 S3(6) ; 0 0 0 0];
    S4b = [0 -S4(3) S4(2) S4(4) ; S4(3) 0 -S4(1) S4(5) ; -S4(2) S4(1) 0 S4(6) ; 0 0 0 0];
    S5b = [0 -S5(3) S5(2) S5(4) ; S5(3) 0 -S5(1) S5(5) ; -S5(2) S5(1) 0 S5(6) ; 0 0 0 0];
    S6b = [0 -S6(3) S6(2) S6(4) ; S6(3) 0 -S6(1) S6(5) ; -S6(2) S6(1) 0 S6(6) ; 0 0 0 0];
    % Converting from world frame to body frame (Eqn 3.75, and just below Eqn 4.16):
    B1b = inv(Tab)*S1b*Tab;
    B2b = inv(Tab)*S2b*Tab;
    B3b = inv(Tab)*S3b*Tab;
    B4b = inv(Tab)*S4b*Tab;
    B5b = inv(Tab)*S5b*Tab;
    B6b = inv(Tab)*S6b*Tab;
    % Write these in the non-bracketed versions as well, [B1] -> B1 etc (3.85 again):
    B1 = [B1b(3,2) B1b(1,3) B1b(2,1) B1b(1,4) B1b(2,4) B1b(3,4)]';
    B2 = [B2b(3,2) B2b(1,3) B2b(2,1) B2b(1,4) B2b(2,4) B2b(3,4)]';
    B3 = [B3b(3,2) B3b(1,3) B3b(2,1) B3b(1,4) B3b(2,4) B3b(3,4)]';
    B4 = [B4b(3,2) B4b(1,3) B4b(2,1) B4b(1,4) B4b(2,4) B4b(3,4)]';
    B5 = [B5b(3,2) B5b(1,3) B5b(2,1) B5b(1,4) B5b(2,4) B5b(3,4)]';
    B6 = [B6b(3,2) B6b(1,3) B6b(2,1) B6b(1,4) B6b(2,4) B6b(3,4)]';
    % The upper-left 3x3 of each bracketed screw axis is the bracketed omega-hat encoding angle (3.85):
    w1b = [B1b(1:3,1:3)];
    w2b = [B2b(1:3,1:3)];
    w3b = [B3b(1:3,1:3)];
    w4b = [B4b(1:3,1:3)];
    w5b = [B5b(1:3,1:3)];
    w6b = [B6b(1:3,1:3)];
    
    % From here on follows the iterative algorithm described above Example 6.1, starting "To modify this algorithm to work with a desired end-effector configuration represented as T_sd...":
    
    thguess = theta0;  % initialize the current guess to the user-supplied value
    lastguess = thguess * 10 + 50;  % arbitrary value far away from the initial guess, so the while loop is entered
    
    while norm(thguess-lastguess) > 1e-3  % this isn't exactly the termination condition the book uses, but loop termination isn't the problem here
      %norm(thguess-lastguess)
      lastguess = thguess;
      % split up the current guess into individual thetas for conciseness below
      t1 = thguess(1); t2 = thguess(2); t3 = thguess(3); t4 = thguess(4); t5 = thguess(5); t6 = thguess(6);
    
      % Exponential coordinate representation of rigid-body motions (take the screw axes above and write them in (R,d) form -- eB1 means exp([B1]theta1), etc.) (3.51, 3.86, 3.87):
      eB1 = [ [eye(3) + sin(t1)*w1b + (1-cos(t1))*(w1b*w1b)] (eye(3)*t1 + (1-cos(t1))*w1b + (t1-sin(t1))*(w1b*w1b))*B1(4:6) ; 0 0 0 1];
      eB2 = [ [eye(3) + sin(t2)*w2b + (1-cos(t2))*(w2b*w2b)] (eye(3)*t2 + (1-cos(t2))*w2b + (t2-sin(t2))*(w2b*w2b))*B2(4:6) ; 0 0 0 1];
      eB3 = [ [eye(3) + sin(t3)*w3b + (1-cos(t3))*(w3b*w3b)] (eye(3)*t3 + (1-cos(t3))*w3b + (t3-sin(t3))*(w3b*w3b))*B3(4:6) ; 0 0 0 1];
      eB4 = [ [eye(3) + sin(t4)*w4b + (1-cos(t4))*(w4b*w4b)] (eye(3)*t4 + (1-cos(t4))*w4b + (t4-sin(t4))*(w4b*w4b))*B4(4:6) ; 0 0 0 1];
      eB5 = [ [eye(3) + sin(t5)*w5b + (1-cos(t5))*(w5b*w5b)] (eye(3)*t5 + (1-cos(t5))*w5b + (t5-sin(t5))*(w5b*w5b))*B5(4:6) ; 0 0 0 1];
      eB6 = [ [eye(3) + sin(t6)*w6b + (1-cos(t6))*(w6b*w6b)] (eye(3)*t6 + (1-cos(t6))*w6b + (t6-sin(t6))*(w6b*w6b))*B6(4:6) ; 0 0 0 1];
    
      % To calculate the Jacobian, we need each of the body screw axes to be transformed by all the joints closer to the root, which is easiest to do using the bracketed form of the screw axes (5.13 and the line above):
      bJ1 = inv(eB6)*inv(eB5)*inv(eB4)*inv(eB3)*inv(eB2)*B1b*eB2*eB3*eB4*eB5*eB6;
      bJ2 = inv(eB6)*inv(eB5)*inv(eB4)*inv(eB3)*B2b*eB3*eB4*eB5*eB6;
      bJ3 = inv(eB6)*inv(eB5)*inv(eB4)*B3b*eB4*eB5*eB6;
      bJ4 = inv(eB6)*inv(eB5)*B4b*eB5*eB6;
      bJ5 = inv(eB6)*B5b*eB6;
      bJ6 = B6b;
      % The Jacobian is the set of column vectors which are the non-bracketed forms of the above (3.85, 5.13, 5.14):
      J1 = [bJ1(3,2) bJ1(1,3) bJ1(2,1) bJ1(1,4) bJ1(2,4) bJ1(3,4)]';
      J2 = [bJ2(3,2) bJ2(1,3) bJ2(2,1) bJ2(1,4) bJ2(2,4) bJ2(3,4)]';
      J3 = [bJ3(3,2) bJ3(1,3) bJ3(2,1) bJ3(1,4) bJ3(2,4) bJ3(3,4)]';
      J4 = [bJ4(3,2) bJ4(1,3) bJ4(2,1) bJ4(1,4) bJ4(2,4) bJ4(3,4)]';
      J5 = [bJ5(3,2) bJ5(1,3) bJ5(2,1) bJ5(1,4) bJ5(2,4) bJ5(3,4)]';
      J6 = [bJ6(3,2) bJ6(1,3) bJ6(2,1) bJ6(1,4) bJ6(2,4) bJ6(3,4)]';
      J = [J1 J2 J3 J4 J5 J6];
    
      % Forward kinematics for the robot's position as a function of the thetas (4.16):
      Tab = M*eB1*eB2*eB3*eB4*eB5*eB6;
    
      % T_bd = T^-1_ab * T_ad (substituting the current guesses for the thetas into the forward kinematics function):
      Tbd = inv(Tab)*T;
    
      % calculate theta and [w] for the matrix logarithm of T_bd (step (b) in the algorithm of section 3.3.3.2, Eqn 3.61):
      if norm(Tbd(1:3,1:3)-eye(3))<1e-5  % case 1: rotation is the identity matrix
         wbd = zeros(3); vbd = Tbd(1:3,4)/norm(Tbd(1:3,4)); thbd = norm(Tbd(1:3,4));
      else  % case 2: "otherwise"
        if trace(Tbd(1:3,1:3))+1 < 1e-5  % case (b): theta = pi; use Eqns 3.58--3.60
           thbd = pi;
           if abs(Tbd(1,1)+1) > 1e-5
	     wbnb = 1/sqrt(2*(1+Tbd(1,1)))*[1+Tbd(1,1); Tbd(2,1); Tbd(3,1)];
           elseif abs(Tbd(2,2)+1) > 1e-5
	     wbnb = 1/sqrt(2*(1+Tbd(2,2)))*[Tbd(1,2); 1+Tbd(2,2); Tbd(3,2)];
           else
	     wbnb = 1/sqrt(2*(1+Tbd(3,3)))*[Tbd(1,3); Tbd(2,3); 1+Tbd(3,3)];
           end
           % now convert that non-bracketed form of w to the bracketed form
           wbd = [0 -wbnb(3) wbnb(2); wbnb(3) 0 -wbnb(1); -wbnb(2) wbnb(1) 0];
        else  % case (c): general case
          thbd = acos((Tbd(1,1)+Tbd(2,2)+Tbd(3,3)-1)/2);
          wbd = 1/2/sin(thbd)*(Tbd(1:3,1:3)-(Tbd(1:3,1:3))');
        end
        % v is calculated according to Eqns 3.91 and 3.92:
        Ginv = eye(3)/thbd - wbd/2 + (1/thbd - cot(thbd/2)/2)*wbd*wbd;
        vbd = Ginv*Tbd(1:3,4);
      end
      % Step 2 of the iterative IK algorithm (above Example 6.1) is to set [Vb] = log(T_bd) = log(T^-1_ab*T_ad)
      Vbb = [wbd*thbd vbd*thbd ; 0 0 0 0];
      %; Now convert to the non-bracketed form Vb of that [Vb]:
      Vb = [Vbb(3,2); Vbb(1,3); Vbb(2,1); Vbb(1:3,4)];
    
      % Update the thetas based on that J and Vb:
      thguess = thguess + pinv(J) * Vb;
    
      % keyboard
    end
    
    theta = thguess;
    
    % Modulate the angles
    thetaMod = mod(theta,2*pi);
    for anglecount = 1:length(thetaMod)
        if thetaMod(anglecount) > pi
            thetaMod(anglecount) = thetaMod(anglecount) - (2*pi);
        else
            thetaMod(anglecount) = thetaMod(anglecount);
        end
    end
    theta = thetaMod;
end