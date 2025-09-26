from knn_lr_gdp import *

#DATA SCINCE VISUALLISATION OF THE DATA

# Plot GDP worldwide over years (deduplicate since same per year)

plt.figure(figsize=(20, 6))
plt.plot(merged_data['year'], merged_data['gdp_worldwide'], marker='o', linestyle='-')
plt.title('Worldwide Real GDP Growth Rate Over Years')
plt.xlabel('Year')
plt.ylabel('Real GDP Worldwide(%)')
plt.show()


# Plot unemployment rate worldwide over years (deduplicate since same per year)
unique_unemp_data = merged_data.drop_duplicates(subset=['year','unemployment_rate_worldwide'])
plt.figure(figsize=(20, 6))
plt.plot(unique_unemp_data['year'],unique_unemp_data['unemployment_rate_worldwide'], marker='o', linestyle='-' )
plt.title('Unemployment rate Worldwide Growth Rate Over Years')
plt.xlabel('Year')
plt.ylabel('Unemployment Rate %')
plt.show()

# Plot migrant numbers over years by country
plt.figure(figsize=(20, 8))
grouped = merged_data.groupby('country')

for country, group in grouped:
    group = group.sort_values('year')  # Ensure sorted by year
    plt.plot(group['year'], group['migrant_numbers'], marker='o', linestyle='-', label=country)
    last_year = group['year'].iloc[-1]
    last_migrants = group['migrant_numbers'].iloc[-1]
    plt.text(last_year + 0.1, last_migrants, country, fontsize=6, verticalalignment='center')

plt.title('Migrant Numbers Over Years by Country')
plt.xlabel('Year')
plt.ylabel('Migrant Numbers')
plt.grid(True)

plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left',fontsize=7, ncol=2, title="Countries")
plt.ticklabel_format(style='plain', axis='y', useOffset=False)
plt.tight_layout()
plt.show()