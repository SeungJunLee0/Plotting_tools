#!/usr/bin/env python3
import os
import re
import ROOT
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep

ROOT.gErrorIgnoreLevel = ROOT.kError
ROOT.gROOT.SetBatch(True)

# ============================================================
# 0) 설정: 입력 폴더 구조 / 출력 폴더
# ============================================================
# 네가 말한 구조: Data/<region>/*.root
DATA_BASE = "Data"

# MC를 따로 분리해뒀으면: MC/<region>/*.root
# 없으면 자동으로 Data/<region>에서 MC도 찾게 함
MC_BASE = "MC" if os.path.isdir("MC") else "Data"

PLOTS_DIR_IN_ROOT = "plots"

out_dir = "Result/compare_plots_mpl_stack_v3_2024"
os.makedirs(out_dir, exist_ok=True)

# ============================================================
# 1) region 폴더 리스트
# ============================================================
regions = [
    "LM_met10_mva_nmtw",
]

# ============================================================
# 2) 파일 리스트 (파일명만)
# ============================================================
data_files = [
    # EGamma0
    "EGamma0_Data_2024C.root",
    "EGamma0_Data_2024D.root",
    "EGamma0_Data_2024E.root",
    "EGamma0_Data_2024F.root",
    "EGamma0_Data_2024G.root",
    "EGamma0_Data_2024H.root",
    "EGamma0_Data_2024I.root",
    "EGamma0_v2_Data_2024I.root",

    # EGamma1
    "EGamma1_Data_2024C.root",
    "EGamma1_Data_2024D.root",
    "EGamma1_Data_2024E.root",
    "EGamma1_Data_2024F.root",
    "EGamma1_Data_2024G.root",
    "EGamma1_Data_2024H.root",
    "EGamma1_Data_2024I.root",
    "EGamma1_v2_Data_2024I.root",

    # Muon0
    "Muon0_Data_2024C.root",
    "Muon0_Data_2024D.root",
    "Muon0_Data_2024E.root",
    "Muon0_Data_2024F.root",
    "Muon0_Data_2024G.root",
    "Muon0_Data_2024H.root",
    "Muon0_Data_2024I.root",
    "Muon0_v2_Data_2024I.root",

    # Muon1
    "Muon1_Data_2024C.root",
    "Muon1_Data_2024D.root",
    "Muon1_Data_2024E.root",
    "Muon1_Data_2024F.root",
    "Muon1_Data_2024G.root",
    "Muon1_Data_2024H.root",
    "Muon1_Data_2024I.root",
    "Muon1_v2_Data_2024I.root",
]

mc_files = [
    # ✅ QCD template (ABCD로 만든 파일)
    "QCD_ABCD.root",

    "DY2E_2J_50_MC_2024.root",
    "DY2M_2J_50_MC_2024.root",
    "DY2T_2J_50_MC_2024.root",

    "s-channel_antitop_MC_2024.root",
    "s-channel_top_MC_2024.root",

    "ST_tW_antitop_di_MC_2024.root",
    "ST_tW_antitop_semi_MC_2024.root",
    "ST_tW_top_di_MC_2024.root",
    "ST_tW_top_semi_MC_2024.root",

    "t-channel_antitop_MC_2024.root",
    "t-channel_top_MC_2024.root",

    "Ttbar_di_MC_2024.root",
    "Ttbar_semi_MC_2024.root",
    "Ttbar_ha_MC_2024.root",

    "WJets_1J_MC_2024.root",
    "WJets_2J_MC_2024.root",
    "WJets_3J_MC_2024.root",
    "WJets_4J_MC_2024.root",

    "WW_MC_2024.root",
    "WWW_4F_MC_2024.root",
    "WZ_MC_2024.root",
    "WZZ_5F_MC_2024.root",

    "ZZ_22_MC_2024.root",
    "ZZ_4_MC_2024.root",
    "ZZ_semi_MC_2024.root",
    "ZZZ_5F_MC_2024.root",
]

# ============================================================
# 3) run(era)별 루미 (fb^-1)
# ============================================================
era_lumi_fb = {
    "2024C": 7.24,
    "2024D": 7.96,
    "2024E": 11.32,
    "2024F": 27.76,
    "2024G": 37.77,
    "2024H": 5.44,
    "2024I": 11.47,
}

def get_era_from_filename(path_or_name):
    base = os.path.basename(path_or_name)
    m = re.search(r"(2024[A-Z])", base)
    return m.group(1) if m else None

# ============================================================
# 4) MC cross section (pb)
#    - QCD_ABCD.root은 템플릿이므로 scale=1로 넣을거라 xsec는 안 씀
# ============================================================
cross_sections_pb = {
    "DY2E_2J_50_MC_2024.root": 6639,
    "DY2M_2J_50_MC_2024.root": 6662,
    "DY2T_2J_50_MC_2024.root": 6630,

    "s-channel_antitop_MC_2024.root": 1.43,
    "s-channel_top_MC_2024.root": 2.278,

    "ST_tW_antitop_di_MC_2024.root": 36.05 * 0.10621,
    "ST_tW_antitop_semi_MC_2024.root": 36.05 * 0.43938,
    "ST_tW_top_di_MC_2024.root": 35.99 * 0.10621,
    "ST_tW_top_semi_MC_2024.root": 35.99 * 0.43938,

    "t-channel_antitop_MC_2024.root": 23.34,
    "t-channel_top_MC_2024.root": 38.6,

    "Ttbar_di_MC_2024.root": 762.1 * 0.10621081,
    "Ttbar_semi_MC_2024.root": 762.1 * 0.43937838,
    "Ttbar_ha_MC_2024.root": 762.1 * (1.0 - 0.10621081 - 0.43937838),

    "WJets_1J_MC_2024.root": 9141,
    "WJets_2J_MC_2024.root": 2931,
    "WJets_3J_MC_2024.root": 864.6,
    "WJets_4J_MC_2024.root": 417.8,

    "WW_MC_2024.root": 11.79,
    "WWW_4F_MC_2024.root": 0.2328,
    "WZ_MC_2024.root": 4.924,
    "WZZ_5F_MC_2024.root": 0.06206,

    "ZZ_22_MC_2024.root": 1.031,
    "ZZ_4_MC_2024.root": 1.39,
    "ZZ_semi_MC_2024.root": 6.788,
    "ZZZ_5F_MC_2024.root": 0.01591,
}

# ============================================================
# 5) 샘플 → 물리 그룹 매핑 / 스택 순서 / 색
# ============================================================
sample_to_group = {
    "QCD_ABCD.root": "QCD",  # ✅ 추가

    "DY2E_2J_50_MC_2024.root": "DY",
    "DY2M_2J_50_MC_2024.root": "DY",
    "DY2T_2J_50_MC_2024.root": "DY",

    "s-channel_antitop_MC_2024.root": "single top s-channel",
    "s-channel_top_MC_2024.root":     "single top s-channel",

    "ST_tW_antitop_di_MC_2024.root":   "single top tW",
    "ST_tW_antitop_semi_MC_2024.root": "single top tW",
    "ST_tW_top_di_MC_2024.root":       "single top tW",
    "ST_tW_top_semi_MC_2024.root":     "single top tW",

    "t-channel_antitop_MC_2024.root":  "single top t-channel",
    "t-channel_top_MC_2024.root":      "single top t-channel",

    "Ttbar_di_MC_2024.root":   "Ttbar",
    "Ttbar_semi_MC_2024.root": "Ttbar",
    "Ttbar_ha_MC_2024.root":   "Ttbar",

    "WJets_1J_MC_2024.root":   "W+jets",
    "WJets_2J_MC_2024.root":   "W+jets",
    "WJets_3J_MC_2024.root":   "W+jets",
    "WJets_4J_MC_2024.root":   "W+jets",

    "WW_MC_2024.root":         "Diboson",
    "WZ_MC_2024.root":         "Diboson",
    "ZZ_22_MC_2024.root":      "Diboson",
    "ZZ_4_MC_2024.root":       "Diboson",
    "ZZ_semi_MC_2024.root":    "Diboson",

    "WWW_4F_MC_2024.root":     "Triboson",
    "WZZ_5F_MC_2024.root":     "Triboson",
    "ZZZ_5F_MC_2024.root":     "Triboson",
}

# (맨 마지막이 맨 위로 쌓임)
group_order = [
    "single top s-channel",  # 신호 맨 위
    "DY",
    "QCD",          # ✅ 추가
    "W+jets",
    "Ttbar",
    "single top t-channel",
    "single top tW",
    "Diboson",
    "Triboson",
]

group_colors = {
    "QCD":                  "#A16207",  # ✅ 추가 (갈색)

    "W+jets":               "#2563EB",
    "Ttbar":                "#F97316",
    "DY":                   "#14B8A6",
    "single top t-channel": "#84CC16",
    "single top tW":        "#8B5CF6",
    "Diboson":              "#64748B",
    "Triboson":             "#4B5563",
    "single top s-channel": "#DC2626",
}

legend_order = [
    "single top t-channel",
    "single top tW",
    "Ttbar",
    "W+jets",
    "DY",
    "QCD",          # ✅ 추가
    "Diboson",
    "Triboson",
    "single top s-channel",
    "Data",
]

# ============================================================
# 6) helper: 히스토 이름 추출
# ============================================================
def get_hist_names(path_to_root):
    f = ROOT.TFile.Open(path_to_root)
    if not f or f.IsZombie():
        raise RuntimeError(f"파일을 열 수 없음: {path_to_root}")

    d = f.GetDirectory(PLOTS_DIR_IN_ROOT)
    if not d:
        f.Close()
        raise RuntimeError(f"'{PLOTS_DIR_IN_ROOT}' 디렉토리가 없음: {path_to_root}")

    d.cd()
    names = [k.GetName() for k in ROOT.gDirectory.GetListOfKeys()]
    f.Close()
    return names

def th1_to_np(h):
    nb = h.GetNbinsX()
    edges = np.array(
        [h.GetBinLowEdge(i) for i in range(1, nb + 1)]
        + [h.GetBinLowEdge(nb) + h.GetBinWidth(nb)],
        dtype=float,
    )
    counts = np.array([h.GetBinContent(i) for i in range(1, nb + 1)], dtype=float)
    return edges, counts

def pick_reference_file(data_paths, mc_paths):
    """hist list 가져올 reference: QCD 파일은 피해서 먼저 잡고, 없으면 data로 fallback"""
    # 1) MC 중 QCD 아닌 파일
    for p in mc_paths:
        if os.path.isfile(p) and os.path.basename(p) != "QCD_ABCD.root":
            return p
    # 2) Data 중 존재하는 파일
    for p in data_paths:
        if os.path.isfile(p):
            return p
    return None

# ============================================================
# 7) region 루프
# ============================================================
for region in regions:
    print("\n==============================")
    print(f"[INFO] REGION = {region}")
    print("==============================")

    region_out_dir = os.path.join(out_dir, region)
    os.makedirs(region_out_dir, exist_ok=True)

    data_dir = os.path.join(DATA_BASE, region)
    mc_dir_1 = os.path.join(MC_BASE, region)  # MC 폴더가 있으면 여기
    mc_dir_2 = os.path.join(DATA_BASE, region)  # 없으면 Data에서 MC도

    if not os.path.isdir(data_dir):
        print(f"[WARN] data dir not found: {data_dir} → skip")
        continue

    # Data/MC 경로 구성
    data_paths = [os.path.join(data_dir, f) for f in data_files]

    # MC는 1순위: MC/<region>, 없으면 Data/<region>
    if os.path.isdir(mc_dir_1):
        mc_paths = [os.path.join(mc_dir_1, f) for f in mc_files]
    else:
        mc_paths = [os.path.join(mc_dir_2, f) for f in mc_files]

    # --- used_eras 계산(존재하는 data 파일에서만) ---
    used_eras = set()
    for p in data_paths:
        if os.path.isfile(p):
            era = get_era_from_filename(p)
            if era in era_lumi_fb:
                used_eras.add(era)

    if not used_eras:
        existing = [p for p in data_paths if os.path.isfile(p)]
        print(f"[WARN] {region}: data 파일(리스트 기준) 발견={len(existing)}개, era 추출 실패 → 스킵")
        continue

    luminosity_fb = sum(era_lumi_fb[e] for e in used_eras)
    luminosity_pb = luminosity_fb * 1000.0
    print("[INFO] used eras:", sorted(used_eras))
    print("[INFO] lumi (fb^-1):", luminosity_fb)

    # --- hist list 가져올 reference 찾기 (QCD 제외 우선) ---
    ref_file = pick_reference_file(data_paths, mc_paths)
    if ref_file is None:
        print(f"[WARN] {region}: reference 파일을 못 찾음 → 스킵")
        continue

    hist_names = get_hist_names(ref_file)

    # mu_*, e_*, combine_* 만
    prefixes = ("mu", "e", "combine")
    hist_names = [h for h in hist_names if any(h.startswith(f"{p}_") for p in prefixes)]

    print(f"[INFO] n_hist to draw = {len(hist_names)} (ref={os.path.basename(ref_file)})")
    if not hist_names:
        print(f"[WARN] {region}: hist_names empty → 스킵")
        continue

    # --- 히스토 루프 ---
    for hname in hist_names:
        print("▶", hname)

        # --------------------
        # Data 합치기
        # --------------------
        data_h = None
        for p in data_paths:
            if not os.path.isfile(p):
                continue
            f = ROOT.TFile.Open(p)
            if not f or f.IsZombie():
                f.Close()
                continue

            h = f.Get(f"{PLOTS_DIR_IN_ROOT}/{hname}")
            if h:
                if data_h is None:
                    data_h = h.Clone(f"data_{region}_{hname}")
                    data_h.SetDirectory(0)
                else:
                    data_h.Add(h)
            f.Close()

        if data_h is None:
            print("   ✖ No data hist, skipping")
            continue

        if data_h.GetDimension() != 1:
            print("   ✖ Skipping non-1D histogram")
            continue

        # --------------------
        # MC 그룹 합산 (QCD 포함)
        # --------------------
        group_sums = {}  # group -> np.array(counts)
        edges_ref = None

        for p in mc_paths:
            if not os.path.isfile(p):
                continue

            base = os.path.basename(p)
            group = sample_to_group.get(base, "other")

            f = ROOT.TFile.Open(p)
            if not f or f.IsZombie():
                f.Close()
                continue

            # ✅ QCD 파일이면 "QCD_D_<hname>__pred" 를 읽는다 + scale=1
            if base == "QCD_ABCD.root":
                qcd_hname = f"QCD_D_{hname}__pred"
                h = f.Get(f"{PLOTS_DIR_IN_ROOT}/{qcd_hname}")
                if not h:
                    f.Close()
                    continue

                tmp = h.Clone(f"tmp_{region}_{hname}_QCD")
                tmp.SetDirectory(0)
                tmp.Sumw2()
                # QCD template은 이미 yield이므로 1로 둠
                tmp.Scale(1.0)

            else:
                h = f.Get(f"{PLOTS_DIR_IN_ROOT}/{hname}")
                if not h:
                    f.Close()
                    continue

                cnt = f.Get(f"{PLOTS_DIR_IN_ROOT}/gen_weight")
                n_evt = cnt.GetSumOfWeights() if cnt else 0.0
                if n_evt <= 0:
                    print(f"   [SKIP] n_evt<=0 ({n_evt}): {base}")
                    f.Close()
                    continue

                xsec = cross_sections_pb.get(base, None)
                if xsec is None:
                    # xsec 없으면 스킵 (실수 방지)
                    f.Close()
                    continue

                scale = (xsec * luminosity_pb) / n_evt

                tmp = h.Clone(f"tmp_{region}_{hname}_{base}")
                tmp.SetDirectory(0)
                tmp.Sumw2()
                tmp.Scale(scale)

            f.Close()

            # edges consistency check (한 번만)
            if edges_ref is None:
                edges_ref, _ = th1_to_np(tmp)

            _, arr = th1_to_np(tmp)

            if group not in group_sums:
                group_sums[group] = arr
            else:
                group_sums[group] += arr

        # 스택 준비
        mc_counts_list = []
        mc_labels = []
        mc_colors = []

        for g in group_order:
            if g in group_sums:
                mc_counts_list.append(group_sums[g])
                mc_labels.append(g)
                mc_colors.append(group_colors.get(g, None))

        if not mc_counts_list:
            print("   ✖ No MC (after scaling), skipping")
            continue

        # --------------------
        # numpy 변환 (data)
        # --------------------
        nb = data_h.GetNbinsX()
        edges = np.array(
            [data_h.GetBinLowEdge(i) for i in range(1, nb + 1)]
            + [data_h.GetBinLowEdge(nb) + data_h.GetBinWidth(nb)],
            dtype=float,
        )

        # 안전: data edges와 mc edges가 다르면 data 기준으로 그림
        # (진짜 다르면 그 전에 histogram definition부터 맞추는 게 정석)
        dat_counts = np.array([data_h.GetBinContent(i) for i in range(1, nb + 1)], dtype=float)
        dat_err = np.sqrt(np.clip(dat_counts, 0, None))
        bin_centers = 0.5 * (edges[:-1] + edges[1:])

        # --------------------
        # Plot
        # --------------------
        plt.style.use(hep.style.CMS)
        fig, (ax, axr) = plt.subplots(
            nrows=2,
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
            figsize=(10, 8),
            dpi=150,
        )

        # MC stack
        hep.histplot(
            mc_counts_list,
            bins=edges,
            stack=True,
            histtype="fill",
            label=mc_labels,
            color=mc_colors,
            ax=ax,
        )

        # Data points
        ax.errorbar(
            bin_centers,
            dat_counts,
            yerr=dat_err,
            fmt="o",
            color="black",
            label="Data",
        )

        ax.set_ylabel("Events / bin")

        hep.cms.label(
            "Private Work",
            data=True,
            lumi=round(luminosity_fb, 2),
            year=2024,
            com=13.6,
            ax=ax,
        )

        # y축 sci offset text를 플롯 안으로
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
        fig.canvas.draw()
        off = ax.yaxis.get_offset_text()
        off_str = off.get_text()
        off.set_visible(False)
        if off_str.strip():
            ax.text(
                0.02, 0.98, off_str,
                transform=ax.transAxes,
                ha="left", va="top",
                fontsize=off.get_size(),
                color="black",
            )

        # Legend reorder
        handles, labels = ax.get_legend_handles_labels()
        label2handle = {lab: h for h, lab in zip(handles, labels)}
        ordered_handles = [label2handle[lab] for lab in legend_order if lab in label2handle]
        ordered_labels = [lab for lab in legend_order if lab in label2handle]
        ax.legend(
            ordered_handles,
            ordered_labels,
            loc="upper right",
            prop={"size": 12},
            handletextpad=0.2,
            labelspacing=0.2,
            columnspacing=0.5,
        )

        # Ratio
        mc_sum = np.sum(mc_counts_list, axis=0)
        mask = mc_sum > 0

        ratio = np.divide(dat_counts, mc_sum, out=np.zeros_like(dat_counts), where=mask)
        ratio_err = np.divide(dat_err, mc_sum, out=np.zeros_like(dat_err), where=mask)

        axr.errorbar(
            bin_centers[mask],
            ratio[mask],
            yerr=ratio_err[mask],
            fmt="o",
            color="black",
        )
        axr.axhline(1.0, color="gray", linestyle="--", linewidth=1)
        axr.set_ylabel("Data/MC")
        axr.set_xlabel(hname)
        axr.set_ylim(0.5, 1.5)

        out_path = os.path.join(region_out_dir, f"stack_{hname}.png")
        fig.savefig(out_path)
        plt.close(fig)

    print(f"[INFO] Done region = {region}")

print("✅ All stacked plots saved in:", out_dir)

