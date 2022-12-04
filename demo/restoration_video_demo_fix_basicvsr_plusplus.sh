python ./demo/restoration_video_demo_fix.py \
./configs/restorers/basicvsr_plusplus/basicvsr_plusplus_c64n7_8x1_600k_reds4.py \
./models/basicvsr_plusplus_c64n7_8x1_600k_reds4_20210217-db622b2f.pth \
"H:\dataset\movie\股疯.mp4" \
./output/股疯_basicvsr_plusplus.mp4 \
--max-seq-len=5 \
--batch-size=2