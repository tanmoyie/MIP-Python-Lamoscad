def compute_stockpiling(z_sor, h_sov, safety_buffer):
    # === Equipment stockpiling from z_sor ===
    df_z = z_sor.reset_index()
    mean_z = df_z.groupby(['level_0', 'level_2'])[0].mean().unstack(fill_value=0)
    equipment_stock = (mean_z * safety_buffer).round().astype(int)
    equipment_stock.reset_index(inplace=True)
    equipment_stock.columns.name = None
    equipment_stock.rename(columns={'level_0': 'Stations'}, inplace=True)

    # Ensure required resource columns exist
    for r in ['m', 'c', 'i']:
        if r not in equipment_stock.columns:
            equipment_stock[r] = 0
    equipment_stock = equipment_stock[['Stations', 'm', 'c', 'i']]

    # === Asset stockpiling from h_sov ===
    df_h = h_sov.reset_index()
    mean_h = df_h.groupby(['level_0', 'level_2'])[0].mean().unstack(fill_value=0)
    asset_stock = (mean_h * safety_buffer).round().astype(int)
    asset_stock.reset_index(inplace=True)
    asset_stock.columns.name = None
    asset_stock.rename(columns={'level_0': 'Stations'}, inplace=True)

    for v in ['helicopter', 'ship', 'icebreaker']:
        if v not in asset_stock.columns:
            asset_stock[v] = 0
    asset_stock = asset_stock[['Stations', 'helicopter', 'ship', 'icebreaker']]

    # === Merge both tables ===
    resource_allocation = asset_stock.merge(equipment_stock, on='Stations', how='inner')
    return resource_allocation