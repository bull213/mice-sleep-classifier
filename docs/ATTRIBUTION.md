# Attribution and external assets

Mouse Sleep Analyzer source is distributed under the [MIT license](../LICENSE).
Its segmentation architecture, morphology, mask measurements, and frequency
feature conventions are based on these upstream projects:

- [KumarLabJax/MouseTracking](https://github.com/KumarLabJax/MouseTracking), with
  [its complete MIT notice](../licenses/MouseTracking_LICENSE.txt).
- [KumarLabJax/MouseSleep](https://github.com/KumarLabJax/MouseSleep), with
  [its complete MIT notice](../licenses/MouseSleep_LICENSE.txt).

Keep these notices when redistributing the source. The project's MIT license
does not relicense third-party dependencies, datasets, or weights.

## Publications and downloads

- Geuther et al., *Robust mouse tracking in complex environments using neural
  networks*, Communications Biology (2019):
  [article](https://doi.org/10.1038/s42003-019-0362-1) and
  [datasets and pretrained models](https://zenodo.org/records/5806397).
- Geuther et al., *High-throughput visual assessment of sleep stages in mice
  using machine learning*, Sleep (2022):
  [article](https://academic.oup.com/sleep/article/45/2/zsab260/6414386) and
  [reference feature data](https://zenodo.org/records/5180680).

Weights, training data, trained classifier artifacts, and experimental
recordings are not included in this source release. Obtain external assets from
the original records and follow their current terms and attribution requirements;
do not assume the code's MIT license applies to them. Locally generated classifier
artifacts are excluded from this repository as well.

The application retrains a modern classifier from the authors' reference
features. It is not an author-distributed pretrained sleep classifier or an
exact reproduction of their validation experiment. Published accuracy is not
claimed for this application's users or their footage.

For the optional annotated albino demonstration used by one verification script,
see [the supplementary-video notice](REFERENCE_NOTICE.md). That video is also
obtained separately.
