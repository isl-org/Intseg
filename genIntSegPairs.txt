%%

% The path to store the synthesized clicks

savepath = ['./train'];

%%
% get the names of all training images
imgset='train';
ids = textread(['./' imgset '.txt'],'%s');
num_img = length(ids);
d_step = 10;
d_margin = 5;
%
for i = 1:num_img
    if ~isdir([savepath '/' ids{i} '/objs'])
        mkdir([savepath '/' ids{i} '/objs']);
    end
    if ~isdir([savepath '/' ids{i} '/ints'])
        mkdir([savepath '/' ids{i} '/ints']);
    end
    imgpath=sprintf('./img/%s.jpg',ids{i});
    img = imread(imgpath);
    objsegpath=sprintf('./inst/%s.mat',ids{i});
    load(objsegpath)
    objseg_img = GTinst.Segmentation;
    sz = size(objseg_img);
    tmp_img = objseg_img;
    % tmp_img(objseg_img==255) = 0;
    num_obj = max(tmp_img(:));
    for j = 1:num_obj
        seg_mask = (tmp_img==j);
        imwrite(seg_mask,[savepath '/' ids{i} '/objs/' num2str(j,'%05d') '.png']);
        
        for k = 1:15 % N_pairs
            pc = zeros(sz); % positive channel
            nc = zeros(sz); % negative channel

            %% positive clicks
            pc_num = randi(10);
            dis_bd = bwdist(1-seg_mask);
            dis_pt = 255*ones(sz);
            for n = 1:pc_num
                [m, ind] = max(rand(sz(1)*sz(2),1).*(dis_bd(:)>d_margin).*(dis_pt(:)>d_step));
                if m ~= 0
                    [r, c] = ind2sub(sz,ind);
                    pc(r,c) = 1;
                    dis_pt = bwdist(pc);
                else
                    break;
                end
            end
            imwrite(uint8(dis_pt),[savepath '/' ids{i} '/ints/' num2str(j,'%03d') '_' num2str(k,'%03d') '_pos.png']);
            
            
            %% negative clicks
            if num_obj > 1
                strat = randi(3);
            else
                strat = randi(2);
            end
            dis_bd = bwdist(seg_mask);
            switch strat
                % Strategy 1
                case 1
                    np_num = randi(15);
                    dis_pt = 255*ones(sz);
                    for n = 1:np_num
                        [m, ind] = max(rand(sz(1)*sz(2),1).*(dis_bd(:)>d_margin).*(dis_pt(:)>d_step));
                        if m ~= 0
                            [r, c] = ind2sub(sz,ind);
                            nc(r,c) = 1;
                            dis_pt = bwdist(nc);
                        else
                            break;
                        end
                    end
                    imwrite(uint8(dis_pt),[savepath '/' ids{i} '/ints/' num2str(j,'%03d') '_' num2str(k,'%03d') '_neg.png']);
                % Strategy 2    
                case 3
                    for tmpj = 1:num_obj
                        if tmpj ~= j
                            np_num = randi(10);
                            tmp_mask = (tmp_img==tmpj);
                            dis_bd = bwdist(1-tmp_mask);
                            dis_pt = 255*ones(sz);
                            for n = 1:np_num
                                [m, ind] = max(rand(sz(1)*sz(2),1).*(dis_bd(:)>d_margin).*(dis_pt(:)>d_step));
                                if m ~= 0
                                    [r, c] = ind2sub(sz,ind);
                                    nc(r,c) = 1;
                                    dis_pt = bwdist(nc);
                                else
                                    break;
                                end
                            end
                        end
                    end
                    imwrite(uint8(dis_pt),[savepath '/' ids{i} '/ints/' num2str(j,'%03d') '_' num2str(k,'%03d') '_neg.png']);
             % Strategy 3    
                case 2
                    np_num = 15;
                    dis_bd = bwdist(seg_mask);
                    sample_region = seg_mask + (dis_bd>=40);
                    % randomly generate the 1st point
                    [m, ind] = max(rand(sz(1)*sz(2),1).*(1-sample_region(:)));
                    [r, c] = ind2sub(sz,ind);
                    nc(r,c) = 1;
                    sample_region = sample_region + nc;
                    for n = 2:np_num
                        dis_bd = bwdist(sample_region);
                        [m, ind] = max(dis_bd(:));
                        if m ~= 0
                            [r, c] = ind2sub(sz,ind);
                            nc(r,c) = 1;
                            sample_region = sample_region + nc;
                        else
                            break;
                        end
                    end
                    dis_pt = bwdist(nc);
                    imwrite(uint8(dis_pt),[savepath '/' ids{i} '/ints/' num2str(j,'%03d') '_' num2str(k,'%03d') '_neg.png']);
            end
            
            %% show points
%             figure,imshow(img);
%             hold on
%             [pos_r,pos_c] = find(pc==1);
%             [neg_r,neg_c] = find(nc==1);
%             plot(pos_c,pos_r,'+g','markersize',8);
%             plot(neg_c,neg_r,'xr','markersize',8);
%             hold off
%             drawnow;pause(0.1);
%             close all;
        end

        
    end
end
