clear all; clc; close all

for pID = 1
    pID

   %DUAL LEVEL
    if pID < 10
        file_name_dctrl  = ['ID0' num2str(pID) '/Dual/ET/ID0' num2str(pID) '_ET_c.xlsx'];
        file_name_d0back = ['ID0' num2str(pID) '/Dual/ET/ID0' num2str(pID) '_ET_0.xlsx'];
        file_name_d1back = ['ID0' num2str(pID) '/Dual/ET/ID0' num2str(pID) '_ET_1.xlsx'];
        file_name_d2back = ['ID0' num2str(pID) '/Dual/ET/ID0' num2str(pID) '_ET_2.xlsx'];
    else
        file_name_dctrl  = ['ID' num2str(pID) '/Dual/ET/ID0' num2str(pID) '_ET_c.xlsx'];
        file_name_d0back = ['ID' num2str(pID) '/Dual/ET/ID0' num2str(pID) '_ET_0.xlsx'];
        file_name_d1back = ['ID' num2str(pID) '/Dual/ET/ID0' num2str(pID) '_ET_1.xlsx'];
        file_name_d2back = ['ID' num2str(pID) '/Dual/ET/ID0' num2str(pID) '_ET_2.xlsx'];    
    end
    
    File_ctrl_draw  = importdata(file_name_dctrl);
    File_0back_draw = importdata(file_name_d0back);
    File_1back_draw = importdata(file_name_d1back);
    File_2back_draw = importdata(file_name_d2back);

    File_dual_ctrl  = File_ctrl_draw.data;
    File_dual_0back = File_0back_draw.data;
    File_dual_1back = File_1back_draw.data;
    File_dual_2back = File_2back_draw.data;
    
    LPD_dual_ctrl  = File_dual_ctrl(:,21);  T_dual_ctrl = File_dual_ctrl(:,4);
    LPD_dual_0back = File_dual_0back(:,21); T_d_0back = File_dual_0back(:,4);
    LPD_dual_1back = File_dual_1back(:,21); T_d_1back = File_dual_1back(:,4);
    LPD_dual_2back = File_dual_2back(:,21); T_d_2back = File_dual_2back(:,4);  
    
    T_dual_0back = T_dual_ctrl(end) + T_d_0back;
    T_dual_1back = T_dual_0back(end) + T_d_1back;
    T_dual_2back = T_dual_1back(end) + T_d_2back;

    %SINGLE LEVEL
    if pID < 10
        file_name_s0back = ['ID0' num2str(pID) '/Single/ET/ID0' num2str(pID) '_ET_0.xlsx'];
        file_name_s1back = ['ID0' num2str(pID) '/Single/ET/ID0' num2str(pID) '_ET_1.xlsx'];
        file_name_s2back = ['ID0' num2str(pID) '/Single/ET/ID0' num2str(pID) '_ET_2.xlsx'];
    else
        file_name_s0back = ['ID' num2str(pID) '/Single/ET/ID0' num2str(pID) '_ET_0.xlsx'];
        file_name_s1back = ['ID' num2str(pID) '/Single/ET/ID0' num2str(pID) '_ET_1.xlsx'];
        file_name_s2back = ['ID' num2str(pID) '/Single/ET/ID0' num2str(pID) '_ET_2.xlsx'];    
    end
    
    File_0back_sraw = importdata(file_name_s0back);
    File_1back_sraw = importdata(file_name_s1back);
    File_2back_sraw = importdata(file_name_s2back);

    File_single_0back = File_0back_sraw.data;
    File_single_1back = File_1back_sraw.data;
    File_single_2back = File_2back_sraw.data;
    
    LPD_single_0back = File_single_0back(:,21); T_single_0back = File_single_0back(:,4);
    LPD_single_1back = File_single_1back(:,21); T_s_1back = File_single_1back(:,4);
    LPD_single_2back = File_single_2back(:,21); T_s_2back = File_single_2back(:,4); 
    
    T_single_1back = T_single_0back(end) + T_s_1back;
    T_single_2back = T_single_1back(end) + T_s_2back;
    
    %DUAL PLOTTING 
    figure;
        hold on; box on; grid on
        plot(T_dual_ctrl, LPD_dual_ctrl,'-')
        plot(T_dual_0back, LPD_dual_0back,'-')
        plot(T_dual_1back, LPD_dual_1back,'-')
        plot(T_dual_2back, LPD_dual_2back, '-')
        xlabel('Time (s)'); ylabel('Pupil Diameter (pixels)')
        title("Participant - " + num2str(pID) + " - Dual")
        legend('Control','0-back','1-back','2-back')
        
    %SINGLE PLOTTING 
    figure;
        hold on; box on; grid on
        plot(T_single_0back, LPD_single_0back,'-')
        plot(T_single_1back, LPD_single_1back,'-')
        plot(T_single_2back, LPD_single_2back, '-')
        xlabel('Time (s)'); ylabel('Pupil Diameter (pixels)')
        title("Participant - " + num2str(pID) + " - Single")
        legend('0-back','1-back','2-back')
    
end


   



