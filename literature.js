const recentPapers = [
  {
    venue: "ICCV 2025 · ALL-IN-ONE RESTORATION",
    title: "FoundIR",
    note: "统一处理多种未知图像退化的强通用恢复模型；本项目使用其官方generalist权重，直接检验恢复视觉是否会消除音频的定位增益。",
    url: "https://openaccess.thecvf.com/content/ICCV2025/html/Cao_FoundIR_Unleashing_Million-scale_Training_Data_to_Advance_Foundation_Models_for_ICCV_2025_paper.html"
  },
  {
    venue: "CVPR 2022 · IMAGE RESTORATION",
    title: "TransWeather",
    note: "代表从去雨、去雾、去雪等视觉恢复入手处理多种恶劣天气；适合作为 Restored-V 强基线，而不是默认等同于定位恢复。",
    url: "https://openaccess.thecvf.com/content/CVPR2022/html/Valanarasu_TransWeather_Transformer-Based_Restoration_of_Images_Degraded_by_Adverse_Weather_Conditions_CVPR_2022_paper.html"
  },
  {
    venue: "WACV 2024 · TASK-AWARE RESTORATION",
    title: "PAIR: Perception Aided Image Restoration",
    note: "指出视觉上令人满意的通用恢复不保证下游感知显著改善，并把语义与深度任务纳入恢复训练；支撑“必须比较任务性能而非只看恢复质量”。",
    url: "https://openaccess.thecvf.com/content/WACV2024/html/Shyam_PAIR_Perception_Aided_Image_Restoration_for_Natural_Driving_Conditions_WACV_2024_paper.html"
  },
  {
    venue: "LANDSCAPE & URBAN PLANNING 2015 · PLACE SOUNDS",
    title: "The Acoustic Summary as a Tool for Representing Urban Sound Environments",
    note: "用一组可感知声事件表征地点，并发现当地居民能够识别本地声学摘要；支持地点声景更像事件集合而非单一 Top 声音。",
    url: "https://doi.org/10.1016/j.landurbplan.2015.08.013"
  },
  {
    venue: "TRANSACTIONS IN GIS 2026 · URBAN SENSING",
    title: "Cross-Modal Urban Sensing",
    note: "实证比较街景、航拍图与环境声音：视觉偏静态几何和材质，声音反映动态及遮挡过程；为选择音频提供城市感知依据，但不能替代定位对照实验。",
    url: "https://onlinelibrary.wiley.com/doi/10.1111/tgis.70246"
  },
  {
    venue: "CVPR 2018 · CHANGING CONDITIONS",
    title: "Benchmarking 6DOF Outdoor Visual Localization in Changing Conditions",
    note: "系统研究昼夜、天气与季节变化下的视觉定位，确立了外观变化与长期鲁棒性问题；但传感证据仍主要来自视觉。",
    url: "https://openaccess.thecvf.com/content_cvpr_2018/html/Sattler_Benchmarking_6DOF_Outdoor_CVPR_2018_paper.html"
  },
  {
    venue: "ICCV 2019 · ILLUMINATION",
    title: "No Fear of the Dark",
    note: "从图像检索角度处理照明变化，代表以视觉表征增强来抵抗退化的路线。",
    url: "https://openaccess.thecvf.com/content_ICCV_2019/papers/Jenicek_No_Fear_of_the_Dark_Image_Retrieval_Under_Varying_Illumination_ICCV_2019_paper.pdf"
  },
  {
    venue: "ICCV 2023 · CROSS-VIEW ROBUSTNESS",
    title: "View Consistent Purification",
    note: "针对移动物体与季节变化净化跨视角特征，说明跨视角定位已有强视觉鲁棒性基线。",
    url: "https://openaccess.thecvf.com/content/ICCV2023/html/Wang_View_Consistent_Purification_for_Accurate_Cross-View_Localization_ICCV_2023_paper.html"
  },
  {
    venue: "WACV 2025 · TEMPORAL SHIFT",
    title: "Temporal Resilience in Geo-Localization",
    note: "研究随时间演化的场景外观，进一步说明退化或域偏移本身不是新问题；我们的缺口必须落在跨传感器可靠性上。",
    url: "https://openaccess.thecvf.com/content/WACV2025W/GeoCV/html/Deuser_Temporal_Resilience_in_Geo-Localization_Adapting_to_the_Continuous_Evolution_of_WACVW_2025_paper.html"
  },
  {
    venue: "2025 · AUDIO GEOLOCATION",
    title: "Audio Geolocation: A Natural Sounds Benchmark",
    note: "证明自然声音包含地理信号，但其独立后验通常较宽；这与我们的条件融合问题互补，而不等同。",
    url: "https://arxiv.org/abs/2505.18726"
  },
  {
    venue: "CVPR 2024 · DATA & SPLIT",
    title: "OpenStreetView-5M",
    note: "强调空间隔离、序列泄露和全球视觉定位的数据偏差；支撑我们对候选库与空间划分合法性的检查。",
    url: "https://openaccess.thecvf.com/content/CVPR2024/papers/Astruc_OpenStreetView-5M_The_Many_Roads_to_Global_Visual_Geolocation_CVPR_2024_paper.pdf"
  },
  {
    venue: "CVPR 2025 · PROBABILISTIC GEOLOCATION",
    title: "Around the World in 80 Timesteps",
    note: "在地球球面上预测位置分布而非单点，明确建模图像可定位性与多峰空间歧义。",
    url: "https://openaccess.thecvf.com/content/CVPR2025/papers/Dufour_Around_the_World_in_80_Timesteps_A_Generative_Approach_to_CVPR_2025_paper.pdf"
  },
  {
    venue: "ICCV 2025 · MULTIMODAL RETRIEVAL",
    title: "MMGeo",
    note: "使用图像加点云、深度或文本查询遥感图库；说明多模态组合检索已有工作，促使我们寻找事件与可靠性层面的更强问题。",
    url: "https://openaccess.thecvf.com/content/ICCV2025/papers/Ji_MMGeo_Multimodal_Compositional_Geo-Localization_for_UAVs_ICCV_2025_paper.pdf"
  },
  {
    venue: "ICCV 2025 · TEXT-TO-MAP",
    title: "CrossText2Loc",
    note: "用自然语言场景描述检索卫星图和OSM；提醒未来文本扩展必须区分场景证据与直接位置泄露。",
    url: "https://openaccess.thecvf.com/content/ICCV2025/papers/Ye_Where_am_I_Cross-View_Geo-localization_with_Natural_Language_Descriptions_ICCV_2025_paper.pdf"
  },
  {
    venue: "CVPR 2026 · NOISY CORRESPONDENCE",
    title: "PAUL",
    note: "研究跨视角配对偏移与不确定性，说明真实系统中遥感正候选完美对齐并非理所当然。",
    url: "https://openaccess.thecvf.com/content/CVPR2026/html/Li_PAUL_Uncertainty-Guided_Partition_and_Augmentation_for_Robust_Cross-View_Geo-Localization_under_CVPR_2026_paper.html"
  },
  {
    venue: "2026 · AUDIOVISUAL GEOLOCATION",
    title: "Interpretable Perception and Reasoning for Audiovisual Geolocation",
    note: "已经使用声学原子消解视觉歧义，因此视觉失败时加入音频不足以单独构成我们的 novelty。",
    url: "https://arxiv.org/abs/2603.05708"
  }
];

const paperStrip = document.querySelector(".paper-strip");
if (paperStrip) {
  const section = document.createElement("section");
  section.className = "recent-literature";
  section.innerHTML = `
    <h3>Recent frontier · 2024–2026</h3>
    <p>这些工作直接影响了今天对任务合法性、概率定位、多模态组合和研究 novelty 的判断；它们是实验设计的对照坐标，而不是装饰性引用。</p>
    <div class="recent-paper-grid">
      ${recentPapers.map(p => `
        <a class="recent-paper" href="${p.url}" target="_blank" rel="noreferrer">
          <small>${p.venue}</small>
          <b>${p.title}</b>
          <span>${p.note}</span>
        </a>
      `).join("")}
    </div>`;
  paperStrip.before(section);
}
