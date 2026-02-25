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
    
    Gx_dual_ctrl   = File_dual_ctrl(:,6);  Gy_dual_ctrl   = File_dual_ctrl(:,7);
    Gx_dual_0back  = File_dual_0back(:,6); Gy_dual_0back  = File_dual_0back(:,7);
    Gx_dual_1back  = File_dual_1back(:,6); Gy_dual_1back  = File_dual_1back(:,7);
    Gx_dual_2back  = File_dual_2back(:,6); Gy_dual_2back  = File_dual_2back(:,7);
    
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
 
    Gx_single_0back  = File_single_0back(:,6); Gy_single_0back  = File_single_0back(:,7);
    Gx_single_1back  = File_single_1back(:,6); Gy_single_1back  = File_single_1back(:,7);
    Gx_single_2back  = File_single_2back(:,6); Gy_single_2back  = File_single_2back(:,7);
    
    %DUAL PLOTTING
        scatter (Gx_dual_ctrl, Gy_dual_ctrl);
        title("Participant - " + num2str(pID) + " - Dual - Control")
        scatter (Gx_dual_0back, Gy_dual_0back);
        title("Participant - " + num2str(pID) + " - Dual - 0Back")
        scatter (Gx_dual_1back, Gy_dual_1back);
        title("Participant - " + num2str(pID) + " - Dual - 1Back")
        scatter (Gx_dual_2back, Gy_dual_2back);
        title("Participant - " + num2str(pID) + " - Dual - 2Back")

    %SINGLE PLOTTING
        scatter (Gx_single_0back, Gy_single_0back);
        title("Participant - " + num2str(pID) + " - Single - 0Back")
        scatter (Gx_single_1back, Gy_single_1back);
        title("Participant - " + num2str(pID) + " - Single - 1Back")
        scatter (Gx_single_2back, Gy_single_2back);
        title("Participant - " + num2str(pID) + " - Single - 2Back")
end

function FigH = scatter(X,Y)
    FigH = figure;
    plot(X, Y,'.')
        xlim([0 1])
        ylim([0 1])
        view([90 90])
        xlabel('X coordinate')
        ylabel('Y coordinate')
end

    
