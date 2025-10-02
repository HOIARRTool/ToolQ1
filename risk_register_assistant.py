import pandas as pd
def get_risk_register_consultation(
        query: str,
        df: pd.DataFrame,
        risk_mitigation_df: pd.DataFrame
):
    """
    ค้นหาข้อมูลอุบัติการณ์ที่ระบุ และดึงข้อมูลที่เกี่ยวข้องออกมา    
    """
    if not query.strip():
        return {"error": "กรุณาป้อนรหัสหรือชื่ออุบัติการณ์ที่ต้องการค้นหา"}

    # --- 1. ค้นหาอุบัติการณ์ที่เกี่ยวข้อง ---
    incident_df = df[
        df['รหัส'].str.contains(query, case=False, na=False) |
        df['ชื่ออุบัติการณ์ความเสี่ยง'].str.contains(query, case=False, na=False)
        ].copy()

    if incident_df.empty:
        return {"error": f"ในช่วงเวลาที่เลือก ยังไม่พบอุบัติการณ์ '{query}'"}

    # --- 2. คำนวณค่าทางสถิติและความเสี่ยง ---
    incident_name = incident_df['ชื่ออุบัติการณ์ความเสี่ยง'].iloc[0]
    incident_code = incident_df['รหัส'].iloc[0]
    total_occurrences = len(incident_df)
    max_impact_level = incident_df['Impact Level'].max()
    frequency_level = incident_df['Frequency Level'].iloc[0]

    highest_risk_row = incident_df.loc[incident_df['Impact Level'] == max_impact_level]
    risk_category = highest_risk_row['Category Color'].iloc[0] if not highest_risk_row.empty else "ไม่ระบุ"
    risk_level_code = f"{max_impact_level}{frequency_level}"

    # --- 2.1. ดึงข้อมูลมาตรการป้องกันและการติดตาม ---
    mitigation_info = risk_mitigation_df[risk_mitigation_df['รหัส'] == incident_code]
    prevention_measure = mitigation_info['มาตรการป้องกันและถ่ายโอนความเสี่ยง'].iloc[
        0] if not mitigation_info.empty else "ไม่มีข้อมูลระบุไว้"
    monitoring_metric = mitigation_info['การติดตาม'].iloc[0] if not mitigation_info.empty else "ไม่มีข้อมูลระบุไว้"

    # --- 3. คืนค่าผลลัพธ์ (ไม่มีการเรียก AI) ---
    return {
        "incident_name": incident_name,
        "incident_code": incident_code,
        "total_occurrences": total_occurrences,
        "max_impact_level": max_impact_level,
        "frequency_level": frequency_level,
        "risk_level_code": risk_level_code,
        "risk_category": risk_category,
        "existing_prevention": prevention_measure,
        "existing_monitor": monitoring_metric,
        "incident_df": incident_df,
    }
