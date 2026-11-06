-- 特征衍生 SQL（场景：industry_finance）
-- 训练端（Spark）和 SQL 产物端共用同一份 SQL 原文
-- 口径：按 part_dt 分区取当日快照；参数 ${bizdate} 为分区日期
-- 注意：本文件不含顶层 WITH，方便内嵌到评分卡 SQL 的 CTE 链中；独立执行时直接运行即可

SELECT
    b.*,

    -- ===== 衍生特征 1：wht_tech_co 是否科技型企业（any_flag）=====
    -- 国家技术创新示范 / 国家高新技术 / 制造业单项冠军 / 专精特新中小企业 /
    -- 科技型中小企业 / 创新型中小企业 / 硬科技赛道 任一='1' 则取 '1'
    CASE
        WHEN TRIM(b.nation_innov_demo_ent_ind) = '1'
          OR TRIM(b.nation_new_tech_ent_ind) = '1'
          OR TRIM(b.manuf_champ_ent_ind) = '1'
          OR TRIM(b.prov_srdi_mid_sml_ent_ind) = '1'
          OR TRIM(b.st_mid_sml_ent_ind) = '1'
          OR TRIM(b.innov_mid_sml_ent_ind) = '1'
          OR TRIM(b.wht_hard_tech) = '1'
        THEN '1'
        ELSE '0'
    END AS wht_tech_co,

    -- ===== 衍生特征 2：rt_rslt_idx 技术流评级（三分类，map_values）=====
    -- TA1/TA2/TA3 → '2'；TB1/TB2 → '1'；其他 → '0'
    CASE
        WHEN TRIM(b.rt_rslt_id) IN ('TA1', 'TA2', 'TA3') THEN '2'
        WHEN TRIM(b.rt_rslt_id) IN ('TB1', 'TB2') THEN '1'
        ELSE '0'
    END AS rt_rslt_idx,

    -- ===== 衍生特征 3：rto_dept_ln 存贷比（存款高于贷款，nested_if）=====
    -- 存款日均≥1000万 且 贷款余额>0 且 存款日均≥3倍贷款余额 → '1'
    CASE
        WHEN CAST(b.dept_bal_avg_y AS DOUBLE) >= 10000000
         AND CAST(b.undisc_ln_bal AS DOUBLE) > 0
         AND CAST(b.dept_bal_avg_y AS DOUBLE) >= CAST(b.undisc_ln_bal AS DOUBLE) * 3
        THEN '1'
        ELSE '0'
    END AS rto_dept_ln,

    -- ===== 衍生特征 4：rto_ln_dept 贷存比（贷款高于存款，nested_if）=====
    -- 贷款余额≥1000万 且 贷款余额>100倍存款日均 → '1'
    CASE
        WHEN CAST(b.undisc_ln_bal AS DOUBLE) >= 10000000
         AND CAST(b.undisc_ln_bal AS DOUBLE) > CAST(b.dept_bal_avg_y AS DOUBLE) * 100
        THEN '1'
        ELSE '0'
    END AS rto_ln_dept,

    -- ===== 衍生特征 5：credit_bal_pct_idx 用信余额占比介于0~10%（nested_if）=====
    -- 用信余额占比>0 且 <0.1 → '1'
    CASE
        WHEN CAST(b.credit_bal_pct AS DOUBLE) > 0
         AND CAST(b.credit_bal_pct AS DOUBLE) < 0.1
        THEN '1'
        ELSE '0'
    END AS credit_bal_pct_idx,

    -- ===== 衍生特征 6：fin_idx 自定义财务指标评分（百分制，nested_if）=====
    -- 资产负债率<65% 且 净利润>0 且 营收≥2500万 且 利润率≥10% → 100
    -- 资产负债率<75% 且 净利润>0 且 营收≥1500万 且 利润率≥8%  → 80
    -- 资产负债率<80% 且 净利润>0 且 营收≥500万              → 60
    -- 否则                                                    → 40
    CASE
        WHEN CAST(b.ast_lia_rt AS DOUBLE) < 0.65
         AND CAST(b.nt_pft AS DOUBLE) > 0
         AND CAST(b.biz_tot_incm AS DOUBLE) >= 25000000
         AND CAST(b.sl_pm_rt AS DOUBLE) >= 0.10
        THEN 100
        WHEN CAST(b.ast_lia_rt AS DOUBLE) < 0.75
         AND CAST(b.nt_pft AS DOUBLE) > 0
         AND CAST(b.biz_tot_incm AS DOUBLE) >= 15000000
         AND CAST(b.sl_pm_rt AS DOUBLE) >= 0.08
        THEN 80
        WHEN CAST(b.ast_lia_rt AS DOUBLE) < 0.80
         AND CAST(b.nt_pft AS DOUBLE) > 0
         AND CAST(b.biz_tot_incm AS DOUBLE) >= 5000000
        THEN 60
        ELSE 40
    END AS fin_idx,

    -- ===== 衍生特征 7：wht_ipo_progress 是否有IPO进程（any_value_nonempty）=====
    -- ipo_progress 非空 → '1'
    CASE
        WHEN b.ipo_progress IS NOT NULL AND TRIM(b.ipo_progress) <> ''
        THEN '1'
        ELSE '0'
    END AS wht_ipo_progress,

    -- ===== 衍生特征 8：wht_rev_growth_rt 是否营业收入增长率超10%（nested_if）=====
    -- rev_growth_rt > 0.1 → '1'
    CASE
        WHEN CAST(b.rev_growth_rt AS DOUBLE) > 0.1
        THEN '1'
        ELSE '0'
    END AS wht_rev_growth_rt,

    -- ===== 衍生特征 9：wht_credit_bal_pct 是否用信占比介于0~10%（nested_if）=====
    -- 用信占比>0 且 ≤0.1 → '1'
    CASE
        WHEN CAST(b.credit_bal_pct AS DOUBLE) > 0
         AND CAST(b.credit_bal_pct AS DOUBLE) <= 0.1
        THEN '1'
        ELSE '0'
    END AS wht_credit_bal_pct

FROM (
    SELECT
        unt_scl_cdt_cd,
        wht_pre_credit, wht_rel_cust, wht_eqt_chain, wht_fund_chain, wht_supp_chain,
        wht_key_ind_cust, credit_cust_ind, entp_st, wht_soe, entp_setup_yr, entp_scl,
        cr_num, industry_code, reg_cap_val, wht_lstd_co, wht_prep_list,
        nation_innov_demo_ent_ind, nation_new_tech_ent_ind, manuf_champ_ent_ind,
        prov_srdi_mid_sml_ent_ind, st_mid_sml_ent_ind, innov_mid_sml_ent_ind,
        wht_hard_tech, wht_above_scale, above_scale_type, wht_cib_chn, chain_code,
        key_cust_ind, rt_rslt_id, wht_credit_pol_pri, wht_credit_pol_mod,
        wht_mkt_access, rtl_shh_rntp_rlt_cib_num_cust,
        otwrd_invst_rntp_rlt_cib_num_cust, supp_chn_core_cust_cnt, fund_chn_rel_cust_cnt,
        wht_major_tender, wht_qk_top30_inv, wht_pe_inv_gt5m, wht_refinance, wht_mna,
        ipo_progress, rev_growth_rt, biz_tot_incm, aset_tot, lia_tot, pft_tot,
        nt_pft, ast_lia_rt, sl_nt_prft_rt, sl_pm_rt, new_reg_entp_ind,
        wht_sim_cust_prod, wht_sim_bal_gt100b, manu_idtfy_flg,
        credit_bal_pct, lmt_amt, in_us_amt, dept_bal_avg_y, undisc_ln_bal,
        check_class_cd, reg_st,
        wht_disc_exec, wht_rel_disc_exec, wht_ser_ileg, wht_bond_warn,
        wht_ser_risk, wht_risk_list, wht_aml_list
    FROM mdm_m01.m01_100_mkt_icf_cust_info
    WHERE part_dt = '${bizdate}'
) b;
