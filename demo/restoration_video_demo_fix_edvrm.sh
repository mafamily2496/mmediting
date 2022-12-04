python ./demo/restoration_video_demo_fix.py \
./configs/restorers/edvr/edvrm_wotsa_x4_g8_600k_reds.py \
./models/edvrm_wotsa_x4_8x4_600k_reds_20200522-0570e567.pth \
"H:\dataset\movie\股疯.mp4" \
./output/股疯_edvrm.mp4 \
--window-size=5 \
--batch-size=2