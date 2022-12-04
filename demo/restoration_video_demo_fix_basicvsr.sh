python ./demo/restoration_video_demo_fix.py \
./configs/restorers/basicvsr/basicvsr_reds4.py \
./models/basicvsr_reds4_20120409-0e599677.pth \
"H:\dataset\movie\蒋介石在黄埔军校十周年庆典上的讲演.mp4" \
./output/蒋介石在黄埔军校十周年庆典上的讲演_basicvsr.mp4 \
--max-seq-len=10 \
--batch-size=8