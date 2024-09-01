#! /bin/bash

function git_show_dirs() {
        for dir_name in `ls $1`
        do
                cur_dir="$1/$dir_name"
                if [ -d $cur_dir ]
                then
			out_dir="${cur_dir}_predict_new"
                     echo $cur_dir
		     echo $out_dir
                    python predict.py $cur_dir $out_dir

                fi
        done
}
git_show_dirs "predict_images2"

