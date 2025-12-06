CoolProp = py.importlib.import_module('CoolProp.CoolProp');

%%Design Parameters calculation: 
data = readmatrix('May.xlsx');  
t_a_in_all = data(:,1);  
rh_in_all  = data(:,2); 
Days = length(t_a_in_all);

efficacy = zeros(Days,1);
for t = 1:Days
 t_a_in = t_a_in_all(t);
    rh_in  = rh_in_all(t)/100;
 
t_out= 35;

C_P= 4.183;
K=1- (C_P/(2493*t_out));
t_w_in=60;

t_wet=CoolProp.HAPropsSI('Twb', 'T', t_a_in+273.15, 'P', 101325, 'R', rh_in)-273.15; 

h_a_in=CoolProp.HAPropsSI('H', 'T', t_a_in+273.15, 'P', 101325, 'R', rh_in)/1000;
h_a_in=round(h_a_in); %inlet air enthalpy.
h_a_out= CoolProp.HAPropsSI('H', 'T', t_a_in+273.15, 'P', 101325, 'R', 1)/1000;
h_a_out=round(h_a_out); %outlet air enthalpy.

%% Relative air flow rate calculation at  
% different air outlet enthalpy.
tolerance = 0.05;
h_a_out_dif= linspace(h_a_in+5,h_a_in+150,30);
lambda= zeros(length(h_a_out_dif),1);

while true

del_t= t_w_in-t_out;
 for i= 1: length(h_a_out_dif)
    lambda(i)=(C_P * del_t)/(K*(h_a_out_dif(i)-h_a_in));


 end


%Relative air flow rate calculation at 
% different relative humidity
%%h_a_out_dif=[150 160 170 180 190 200]

m_w= 1e5 * 1000 / 3600;
Fill_area= pi* (60)^2;
Height= 165;
drag=15;
rh_out=[1,0.98,0.96, 0.94,0.92,0.90];
t_a_out= zeros (length(rh_out),1);
p_v_in=CoolProp.PropsSI('P','T',t_a_in+273.15,'Q',0,'Water')* rh_in;

P=101325;
rho_a_in= 1/CoolProp.HAPropsSI('Vha', 'T', t_a_in+273, 'P', P, 'R', rh_in);
rho_a_out=zeros(length(rh_out),1);

temp_lambda= zeros(length(h_a_out_dif),length(rho_a_out) );

for i=1:length(rh_out)
    for j= 1: length(h_a_out_dif)
        t_a_out(i)= CoolProp.HAPropsSI('T', 'H', h_a_out_dif(j)*1e3, 'P', P, 'R', rh_out(i))- 273.15;
        
        rho_a_out(i)= 1/CoolProp.HAPropsSI('Vha', 'T', t_a_out(i)+273, 'P', P, 'R', rh_out(i));
        if(((rho_a_in)^2 -(rho_a_out(i))^2)>0)
            temp_lambda(j,i)= (Fill_area/m_w)* sqrt((Height*((rho_a_in)^2 -(rho_a_out(i))^2)* 9.8)/drag );
        else
            temp_lambda(j,i)= 0;
        end
    end

end


lambda_100=temp_lambda(:,1);
lambda_96=temp_lambda(:,2);
lambda_92=temp_lambda(:,3);
lambda_88=temp_lambda(:,4);
lambda_84=temp_lambda(:,5);
lambda_80=temp_lambda(:,6);
%hold on
%plot(h_a_out_dif,lambda,'c','LineWidth',2)
%plot(h_a_out_dif,lambda_100,'b','LineWidth',2)
%plot(h_a_out_dif,lambda_96,'g','LineWidth',2)
%plot(h_a_out_dif,lambda_92,'y','LineWidth',2)
%plot(h_a_out_dif,lambda_88,'r','LineWidth',2)
%plot(h_a_out_dif,lambda_84,'m','LineWidth',2)
%plot(h_a_out_dif,lambda_80,'k','LineWidth',2)
%xlabel('Outlet Air Enthalpy');
%ylabel('Relative air flow rate');
%title('Relative air flow rate vs Outlet Air Enthalpy');
%legend('A', 'w=100%','w=96%','w=92%', 'w=88%','w=84%','w=80%','Location','best')
x=h_a_out_dif;
y=lambda;
y100=lambda_100;
y96=lambda_96;
y92=lambda_92;
y88=lambda_88;
y84=lambda_84;
y80=lambda_80;
find_intersections= @(y1,y2) interp1(y1-y2,x,0);
x_100=find_intersections(y,y100);
x_96=find_intersections(y,y96);
x_92=find_intersections(y,y92);
x_88=find_intersections(y,y88);
x_84=find_intersections(y,y84);
x_80=find_intersections(y,y80);
%outlet air temperature
h_a_out_original=[x_100;x_96;x_92;x_88;x_84;x_80];
t_a_out_original= [];
rh_out_new=[];
lambda_original=[];
for i=1 : length(h_a_out_original)
    if ~ isnan (h_a_out_original(i))
        
temp=CoolProp.HAPropsSI('T', 'H', h_a_out_original(i)*1000, 'P', 101325, 'R', rh_out(i))-273.15;
t_a_out_original=[t_a_out_original;temp];

lambda_original=[lambda_original;interp1(x,y, h_a_out_original(i))];
 rh_out_new=[rh_out_new;rh_out(i)];
    end
end


    if t_a_in >= 0 && t_a_in <= 19
    t_w_out = 27:2:35;
elseif t_a_in > 19 && t_a_in <= 23
    t_w_out = 29:2:37;
elseif t_a_in > 23 && t_a_in <= 26.5
    t_w_out = 31:2:39;
elseif t_a_in > 26.5 && t_a_in <= 29.5
    t_w_out = 33:2:41;
    elseif t_a_in > 29.5 && t_a_in <= 33.5
    t_w_out = 35:2:43; % fallback for higher values
    elseif t_a_in > 33.5 && t_a_in <= 37.5
         t_w_out = 38:2:46;
    else 
        t_w_out = 40:2:48 ;
end

omega_in=py.CoolProp.CoolProp.HAPropsSI('W','T',t_a_in+273.15,'P',101325,'RH',rh_in);
p1_unsat=py.CoolProp.CoolProp.PropsSI('P','T', t_a_in+273.15,'Q',0,'Water')*rh_in;
p1_double_prime=py.CoolProp.CoolProp.PropsSI('P','T', t_w_in+273.15,'Q',0,'Water');
omega_out=zeros(length(t_a_out_original),1);
p2_unsat=zeros(length(t_a_out_original),1);
%t_a_out_original_pro=zeros(length(t_a_out_original),1);
temp_t_w_out=zeros(length(t_a_out_original),length(t_w_out) );

for k= 1: length(t_w_out)
    for j=1: length(t_a_out_original)
%p1_double_prime=py.CoolProp.CoolProp.PropsSI('P','T', t_w_in+273.15,'Q',0,'Water');
p2_double_prime(k)=py.CoolProp.CoolProp.PropsSI('P','T', t_w_out(k)+273.15,'Q',0,'Water');
%p1_unsat=py.CoolProp.CoolProp.PropsSI('P','T', t_a_in+273.15,'Q',0,'Water')*rh_in;
p_m_double_prime(k)= py.CoolProp.CoolProp.PropsSI('P','T',((t_w_in+t_w_out(k))/2)+273.15,'Q',0,'Water');
del_p_double_prime(k)=0.25* (p1_double_prime+p2_double_prime(k)- 2*p_m_double_prime(k));

%p2_unsat=zeros(length(t_a_out_original),1);
%t_a_out_original_pro=zeros(length(p2_unsat),1);
%for j=1: length(t_a_out_original)
    omega_out(j)=py.CoolProp.CoolProp.HAPropsSI('W','T',t_a_out_original(j)+273.15,'P',101325,'RH',rh_out(j));
p2_unsat(j)= py.CoolProp.CoolProp.PropsSI('P','T', t_a_out_original(j)+273.15,'Q',0,'Water')*rh_out(j);



temp_t_w_out(j,k)=t_a_in+13.7*10^4*(omega_out(j)-omega_in)*((t_w_in+t_w_out(k)-t_a_in-t_a_out_original(j))/(p1_double_prime+p2_double_prime(k)-p1_unsat-p2_unsat(j)-2*del_p_double_prime(k)));
    end
end
t_w_32=temp_t_w_out(:,1);
t_w_34=temp_t_w_out(:,2);
t_w_36=temp_t_w_out(:,3);
t_w_38=temp_t_w_out(:,4);
t_w_40=temp_t_w_out(:,5);



% Stack downward curves column-wise → 6×5 matrix
y_down_all = [t_w_32, t_w_34, t_w_36, t_w_38, t_w_40];

% Fit the upward curve once
p_up = polyfit(lambda_original, t_a_out_original, 1);  % [slope, intercept]
m1 = p_up(1);
c1 = p_up(2);

% Prepare output vectors (5×1)
lambda_original_new = zeros(5, 1);       % x-coordinates of intersections
t_a_out_original_new = zeros(5, 1);      % y-coordinates of intersections

% Loop over each column (each downward curve)
for i = 1:5
    y_down = y_down_all(:, i);                       % Get one downward curve
    p_down = polyfit(lambda_original, y_down, 1);    % Fit a line
    m2 = p_down(1);
    c2 = p_down(2);

    % Find intersection point
    x_int = (c2 - c1) / (m1 - m2);
    y_int = m1 * x_int + c1;

    % Store result
    lambda_original_new(i) = x_int;
    t_a_out_original_new(i) = y_int;
end

p_up_again = polyfit(rh_out, t_a_out_original, 1);  % [slope, intercept]
m1_again = p_up_again(1);
c1_again = p_up_again(2);

rh_new_out = zeros(5, 1); 
for i = 1:5
    y_down_again = y_down_all(:, i);                       % Get one downward curve
    p_down_again = polyfit(rh_out, y_down_again, 1);    % Fit a line
    m2_again = p_down_again(1);
    c2_again = p_down_again(2);

    % Find intersection point
   rh_new_out(i) = min((c2_again - c1_again) / (m1_again - m2_again),1);
   
end


%clf
%hold on
%plot(lambda_original,t_a_out_original,'c','LineWidth',2)
%plot(lambda_original,t_w_32,'r','LineWidth',2)
%plot(lambda_original,t_w_34,'m','LineWidth',2)
%plot(lambda_original,t_w_36,'b','LineWidth',2)
%plot(lambda_original,t_w_38,'g','LineWidth',2)
%plot(lambda_original,t_w_40,'y','LineWidth',2)

%3rd part
d= 0.05; B=101325;
for i = 1: length(t_a_out_original_new)
    T_mean(i)= (t_a_out_original_new(i)+273.15+t_a_in+273.15)/2;
Diffusion_coeff(i)= (1.75*10^-5)/B *(T_mean(i)/273)^0.8 ;
humid_den(i)=1/CoolProp.HAPropsSI('Vha', 'T', T_mean(i), 'P', B, 'R', rh_in);
dynamic_visc(i)=CoolProp.HAPropsSI('mu', 'T',T_mean(i), 'P', B,'RH',rh_in);
kinematic_visc(i)=dynamic_visc(i)/humid_den(i);
air_flow_rate(i)=lambda_original_new(i)* m_w;
air_flow_velocity(i)=air_flow_rate(i)/(Fill_area * humid_den(i));
film_velocity(i)= ((0.24-0.23)/10)*((t_w_in+t_w_out(i)- 40)) + 0.23;
air_relative_velo(i)= air_flow_velocity(i)+ film_velocity(i);
reynold_no(i)= air_relative_velo(i)* d/ kinematic_visc(i);
  if reynold_no(i) < 1e4
        beta_p(i) = (0.0008 * (reynold_no(i))^1.18 * Diffusion_coeff(i)) / d;
    else
        beta_p(i) = (0.028 * (reynold_no(i))^0.8 * Diffusion_coeff(i)) /d;
  end
  
    del_p1(i)= p1_double_prime-p2_unsat(i);
    del_p2(i)=p2_double_prime(i)-p1_unsat;
    del_p_cp(i)= (del_p1(i)-del_p2(i))/ log ((del_p1(i)-del_p_double_prime(i))/(del_p2(i)-del_p_double_prime(i)));
   omega_out_new(i)= CoolProp.HAPropsSI('W','T',t_a_out_original_new(i)+273.15,'P',101325,'RH',rh_new_out(i));
 F(i)=(lambda_original_new(i)* m_w * (omega_out_new(i) - omega_in)) / (beta_p(i) * del_p_cp(i) );

end 
%plot(t_w_out, F,'m-o','LineWidth',2)
F_known=1.5e6;
t_w_exact= interp1( F,t_w_out, F_known,'linear', 'extrap');
if abs(t_w_exact - t_out) < tolerance
       break;
   end

 t_out = t_w_exact;

end
t_w_pain=t_w_exact;
efficacy(t)=((t_w_in-t_w_pain)/(t_w_in-t_wet))* 100
end
figure;
bar(1:Days, efficacy ,'FaceColor', 'r');
xlabel('Day');
ylabel('Cooling Efficiency (%)');
title('Cooling Efficiency for May');
grid on;
%lambda_exact= interp1(t_w_out,lambda_original_new,t_w_pain, 'linear','extrap');
%t_a_out_exact= interp1(t_w_out,t_a_out_original_new,t_w_pain,'linear','extrap');
%rh_exact=interp1(t_w_out,rh_new_out,t_w_pain,'linear','extrap')
%t_make_up=30;
%m_a= m_w* lambda_exact
%omega_exact=CoolProp.HAPropsSI('W','T',t_a_out_exact+273.15,'P',101325,'RH',rh_exact);
%h_a_out_exact=CoolProp.HAPropsSI('H', 'T', t_a_out_exact+273.15, 'P', 101325, 'R', rh_exact)/1000;
%h_make_up=CoolProp.PropsSI('H','T',t_make_up+273.15,'Q',0,'Water' )/1000;
%m_make_up= m_a * (omega_exact-omega_in);
%Q_rej= (m_a*(h_a_out_exact- h_a_in)- m_make_up* h_make_up)/1000;
%eta= Q_rej/2.79e3 ;