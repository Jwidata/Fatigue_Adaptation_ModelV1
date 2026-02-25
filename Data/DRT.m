clear all; clc; close all
for pID = 1
    pID
    %DUAL LEVEL
    if pID < 10
        file_name_dctrl  = ['ID0' num2str(pID) '/Dual/DRT/ID0' num2str(pID) '_DRT_c.xlsx'];
        file_name_d0back = ['ID0' num2str(pID) '/Dual/DRT/ID0' num2str(pID) '_DRT_0.xlsx'];
        file_name_d1back = ['ID0' num2str(pID) '/Dual/DRT/ID0' num2str(pID) '_DRT_1.xlsx'];
        file_name_d2back = ['ID0' num2str(pID) '/Dual/DRT/ID0' num2str(pID) '_DRT_2.xlsx'];
    else
        file_name_dctrl  = ['ID' num2str(pID) '/Dual/DRT/ID0' num2str(pID) '_DRT_c.xlsx'];
        file_name_d0back = ['ID' num2str(pID) '/Dual/DRT/ID0' num2str(pID) '_DRT_0.xlsx'];
        file_name_d1back = ['ID' num2str(pID) '/Dual/DRT/ID0' num2str(pID) '_DRT_1.xlsx'];
        file_name_d2back = ['ID' num2str(pID) '/Dual/DRT/ID0' num2str(pID) '_DRT_2.xlsx'];    
    end
    
    File_ctrl_draw  = importdata(file_name_dctrl);
    File_0back_draw = importdata(file_name_d0back);
    File_1back_draw = importdata(file_name_d1back);
    File_2back_draw = importdata(file_name_d2back);

    File_dual_ctrl  = File_ctrl_draw.data;
    File_dual_0back = File_0back_draw.data;
    File_dual_1back = File_1back_draw.data;
    File_dual_2back = File_2back_draw.data;
     
    [T_ctrl, RT_ctrl]   = drtClean(File_dual_ctrl(:,3), File_dual_ctrl(:,5), File_dual_ctrl(:,6)); 
    [T_0back, RT_0back] = drtClean(File_dual_0back(:,3), File_dual_0back(:,5), File_dual_0back(:,6)); 
    [T_1back, RT_1back] = drtClean(File_dual_1back(:,3), File_dual_1back(:,5), File_dual_1back(:,6)); 
    [T_2back, RT_2back] = drtClean(File_dual_2back(:,3), File_dual_2back(:,5), File_dual_2back(:,6)); 

    T_0back = T_ctrl(end) + T_0back;
    T_1back = T_0back(end) + T_1back;
    T_2back = T_1back(end) + T_2back;

    figure;
    hold on; box on; grid on
    plot(T_ctrl, RT_ctrl,'-')
    plot(T_0back, RT_0back,'-')
    plot(T_1back, RT_1back,'-')
    plot(T_2back, RT_2back, '-')
    xlabel('Time (s)'); ylabel('Response Time (ms)')
    title("Participant - " + num2str(pID) + " - Dual")
    legend('Control','0-back','1-back','2-back')

end

function [T, RT] = drtClean(drtTime, drtClicks, drtRT)
    idx = (drtClicks==1 & drtRT>100 & drtRT<2500);
    T = drtTime(idx,:)/1000; % in seconds
    RT = drtRT(idx,:);
end
