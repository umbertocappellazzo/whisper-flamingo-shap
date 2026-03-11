## Shapley Value computations for Whisper-Flamingo

This repository contains the code to compute the audio/video SHAP contributions for the Whisper-Flamingo model. For more details, please refer to our [`paper`](??). 

---

## Requirements

To setup the environment and download the test files for LRS3, please refer to the official [`Whisper-Flamingo repository`](https://github.com/roudimit/whisper-flamingo) with all the details. Once this is done, make sure to install the **shap** and **wandb** libraries: ```pip install shap wandb==0.15.12```. In addition to this, download the Whisper-Flamingo checkpoint we used in our manuscript. The ckpt we used is the audio-visual Large-V2 noisy model pre-trained on LRS3 + VoxCeleb2 (En). The ckpt name is `whisper-flamingo_en_large_vc2_noisy.pt`.  

## Compute the global A/V-SHAP Contributions.

To compute the A/V-SHAP contributions, run the command as below: 

```Shell
python -u whisper_decode_video_shap.py --exp-name [exp_name] --lang en --model-type large-v2 --modalities avsr \
--use_av_hubert_encoder 1 --av_fusion separate --checkpoint-path [path_to_ckpt/whisper-flamingo_en_large_vc2_noisy.pt] \
--av-hubert-ckpt /ucappell/whisper-flamingo-shap/models/large_noise_pt_noise_ft_433h.pt --whisper-path [path_to_whisper_ckpt] \
--num-samples-shap [num_samples_shap] --shap-alg [shap_alg] --wandb-project [wandb_project] --output-path [output_path] \
--noise-fn [path_to_noise] --noise-snr 10
```

 <details open>
  <summary><strong>Main Arguments</strong></summary>

- `exp-name`: The experiment name.
- `checkpoint-path`: The path to the Whisper-Flamingo checkpoint.
- `whisper-path`: The path to the Whisper checkpoint.
- `num-samples-shap`: The number of coalitions to sample.
- `shap-alg`: The algorithm from the shap library to compute the shapley matrix. Choices: [`sampling`, `permutation`].
- `wandb-project`: Name of the wandb project to track the results.
- `output-path`: The path to save the SHAP values for further analyses. This folder must be created beforehand!
- `noise-fn`: The path to the folder containing noise manifest files ([path_to_noise]/{valid,test}.tsv).
- `noise-snr`: The SNR level of acoustic noise to test on.


</details>


---

## 🔖 Citation

If you find our work useful, please cite:

```bibtex
@article{cappellazzo2026ODrSHAPAV,
  title={Dr. SHAP-AV: Decoding Relative Modality Contributions via Shapley Attribution in Audio-Visual Speech Recognition},
  author={Umberto, Cappellazzo and Stavros, Petridis and Maja, Pantic},
  journal={arXiv preprint arXiv:?},
  year={2026}
}
```

---

## 🙏 Acknowledgements

- Our code relies on [Whisper-Flamingo](https://github.com/roudimit/whisper-flamingo)

---

## 📧 Contact

For questions and discussions, please:
- Open an issue on GitHub
- Email: umbertocappellazzo@gmail.com
- Visit our [project page](https://umbertocappellazzo.github.io/Omni-AVSR/) and our [preprint](https://arxiv.org/abs/2511.07253)

---




