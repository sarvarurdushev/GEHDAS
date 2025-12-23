"""
🌌 GEDHAS - Galactic Exoplanet Discovery & Habitability Assessment System
Streamlit Dashboard for Exoplanet Analysis
"""

import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random
import math
from datetime import datetime, timedelta
import os

# Set page configuration
st.set_page_config(
    page_title="GEDHAS - Exoplanet Discovery System",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for space theme
st.markdown("""
<style>
    .stApp {
        background:  linear-gradient(135deg, #0a0a1a 0%, #1a1a2e 50%, #16213e 100%);
    }
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        border:  1px solid #00ff88;
    }
    .metric-card {
        background: rgba(26, 26, 46, 0.8);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #66aaff;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DATABASE FUNCTIONS
# ============================================================================

@st.cache_resource
def init_database():
    """Initialize and populate the database (cached)"""
    conn = sqlite3.connect(':memory:', check_same_thread=False)
    create_schema(conn)
    populate_data(conn)
    return conn

def create_schema(conn):
    """Create database schema"""
    cursor = conn.cursor()
    
    # STAR_SYSTEMS table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS STAR_SYSTEMS (
            star_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            spectral_type TEXT NOT NULL,
            luminosity REAL NOT NULL,
            temperature INTEGER NOT NULL,
            distance_ly REAL NOT NULL,
            age_gyr REAL NOT NULL,
            mass_solar REAL NOT NULL,
            ra_deg REAL NOT NULL,
            dec_deg REAL NOT NULL,
            galactic_quadrant TEXT NOT NULL,
            metallicity REAL NOT NULL
        )
    ''')
    
    # EXOPLANETS table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS EXOPLANETS (
            planet_id INTEGER PRIMARY KEY AUTOINCREMENT,
            star_id INTEGER NOT NULL,
            name TEXT NOT NULL UNIQUE,
            mass_earth REAL NOT NULL,
            radius_earth REAL NOT NULL,
            orbital_period_days REAL NOT NULL,
            semi_major_axis_au REAL NOT NULL,
            eccentricity REAL NOT NULL,
            equilibrium_temp_k REAL NOT NULL,
            surface_gravity_earth REAL NOT NULL,
            density_gcc REAL NOT NULL,
            planet_type TEXT NOT NULL,
            discovery_date DATE NOT NULL,
            confirmed BOOLEAN NOT NULL DEFAULT 1,
            FOREIGN KEY (star_id) REFERENCES STAR_SYSTEMS(star_id)
        )
    ''')
    
    # DISCOVERY_MISSIONS table
    cursor. execute('''
        CREATE TABLE IF NOT EXISTS DISCOVERY_MISSIONS (
            mission_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            organization TEXT NOT NULL,
            mission_type TEXT NOT NULL,
            detection_method TEXT NOT NULL,
            launch_date DATE NOT NULL,
            end_date DATE,
            status TEXT NOT NULL,
            total_discoveries INTEGER DEFAULT 0,
            sensitivity_rating INTEGER NOT NULL
        )
    ''')
    
    # HABITABILITY_SCORES table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS HABITABILITY_SCORES (
            score_id INTEGER PRIMARY KEY AUTOINCREMENT,
            planet_id INTEGER NOT NULL UNIQUE,
            esi_score REAL NOT NULL,
            hz_status TEXT NOT NULL,
            water_probability REAL NOT NULL,
            atmosphere_rating INTEGER NOT NULL,
            magnetic_field_probability REAL NOT NULL,
            tidal_lock_probability REAL NOT NULL,
            habitability_class TEXT NOT NULL,
            study_priority INTEGER NOT NULL,
            FOREIGN KEY (planet_id) REFERENCES EXOPLANETS(planet_id)
        )
    ''')
    
    # ATMOSPHERIC_ANALYSIS table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ATMOSPHERIC_ANALYSIS (
            analysis_id INTEGER PRIMARY KEY AUTOINCREMENT,
            planet_id INTEGER NOT NULL UNIQUE,
            analysis_date DATE NOT NULL,
            hydrogen_pct REAL DEFAULT 0,
            helium_pct REAL DEFAULT 0,
            nitrogen_pct REAL DEFAULT 0,
            oxygen_pct REAL DEFAULT 0,
            carbon_dioxide_pct REAL DEFAULT 0,
            methane_pct REAL DEFAULT 0,
            water_vapor_pct REAL DEFAULT 0,
            biosignature_detected BOOLEAN DEFAULT 0,
            FOREIGN KEY (planet_id) REFERENCES EXOPLANETS(planet_id)
        )
    ''')
    
    conn.commit()

def calculate_habitable_zone(luminosity, temp_k):
    """Calculate habitable zone boundaries"""
    t_star = temp_k - 5780
    s_eff_inner = 1.0146 + 8.1884e-5 * t_star + 1.9394e-9 * t_star**2
    s_eff_outer = 0.3507 + 5.9578e-5 * t_star + 1.6707e-9 * t_star**2
    inner_hz = math.sqrt(luminosity / s_eff_inner)
    outer_hz = math.sqrt(luminosity / s_eff_outer)
    return (inner_hz, outer_hz)

def calculate_esi(radius_earth, mass_earth, temp_k, escape_vel_ratio=1.0):
    """Calculate Earth Similarity Index"""
    r_ref, m_ref, t_ref, v_ref = 1.0, 1.0, 288.0, 1.0
    w_r, w_m, w_t, w_v = 0.57, 1.07, 5.58, 0.70
    
    esi_r = (1 - abs((radius_earth - r_ref) / (radius_earth + r_ref))) ** w_r
    esi_m = (1 - abs((mass_earth - m_ref) / (mass_earth + m_ref))) ** w_m
    esi_t = (1 - abs((temp_k - t_ref) / (temp_k + t_ref))) ** w_t
    esi_v = (1 - abs((escape_vel_ratio - v_ref) / (escape_vel_ratio + v_ref))) ** w_v
    
    return min(max((esi_r * esi_m * esi_t * esi_v) ** 0.25, 0), 1)

def classify_planet_type(mass_earth, radius_earth, temp_k):
    """Classify planet type"""
    if mass_earth < 0.1: 
        return "Dwarf Planet"
    elif mass_earth < 2 and radius_earth < 1.5:
        if 200 < temp_k < 350:
            return "Potentially Habitable Terrestrial"
        elif temp_k >= 350:
            return "Hot Terrestrial"
        else:
            return "Cold Terrestrial"
    elif mass_earth < 10 and radius_earth < 2.5:
        if temp_k >= 350:
            return "Hot Super-Earth"
        elif 200 < temp_k < 350:
            return "Temperate Super-Earth"
        else: 
            return "Cold Super-Earth"
    elif mass_earth < 20: 
        return "Mini-Neptune"
    elif mass_earth < 100:
        return "Neptune-like"
    elif mass_earth < 500:
        return "Sub-Jupiter"
    elif mass_earth < 3000:
        return "Hot Jupiter" if temp_k > 1000 else "Jupiter-like"
    else:
        return "Super-Jupiter"

def populate_data(conn):
    """Populate database with mock data"""
    random.seed(42)
    np.random.seed(42)
    cursor = conn.cursor()
    
    # Generate stars
    spectral_data = [
        ('M', 0.45, (2400, 3700), (0.001, 0.08), (0.08, 0.45)),
        ('K', 0.25, (3700, 5200), (0.08, 0.6), (0.45, 0.8)),
        ('G', 0.15, (5200, 6000), (0.6, 1.5), (0.8, 1.04)),
        ('F', 0.08, (6000, 7500), (1.5, 5), (1.04, 1.4)),
        ('A', 0.04, (7500, 10000), (5, 25), (1.4, 2.1)),
    ]
    
    greek = ['Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon', 'Zeta', 'Eta', 'Theta']
    constellations = ['Novarum', 'Sideris', 'Caelum', 'Astralis', 'Stellara', 'Cosmicus']
    quadrants = ['Alpha', 'Beta', 'Gamma', 'Delta']
    
    star_ids = []
    for i in range(100):
        rand = random.random()
        cumulative = 0
        for spec_type, prob, temp_range, lum_range, mass_range in spectral_data:
            cumulative += prob
            if rand <= cumulative:
                spectral_type = spec_type + str(random.randint(0, 9))
                temperature = random.randint(temp_range[0], temp_range[1])
                luminosity = random.uniform(lum_range[0], lum_range[1])
                mass = random.uniform(mass_range[0], mass_range[1])
                break
        
        name = f"{random.choice(greek)} {random.choice(constellations)} {random.randint(1, 999)}"
        distance = random.expovariate(1/100) + 10
        age = random.uniform(1, 10)
        
        try:
            cursor. execute('''
                INSERT INTO STAR_SYSTEMS (name, spectral_type, luminosity, temperature, 
                    distance_ly, age_gyr, mass_solar, ra_deg, dec_deg, galactic_quadrant, metallicity)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (name, spectral_type, round(luminosity, 4), temperature,
                  round(distance, 2), round(age, 2), round(mass, 3),
                  round(random.uniform(0, 360), 4), round(random.uniform(-90, 90), 4),
                  random.choice(quadrants), round(random.gauss(0, 0.3), 3)))
            star_ids.append(cursor.lastrowid)
        except: 
            continue
    
    # Generate planets
    planet_designators = ['b', 'c', 'd', 'e', 'f', 'g']
    planet_ids = []
    
    for star_id in star_ids:
        cursor.execute('SELECT name, luminosity, temperature, mass_solar FROM STAR_SYSTEMS WHERE star_id = ?', (star_id,))
        star = cursor.fetchone()
        star_name, luminosity, temp_k, star_mass = star
        
        hz_inner, hz_outer = calculate_habitable_zone(luminosity, temp_k)
        num_planets = min(np.random.poisson(3), 6)
        
        for j in range(num_planets):
            planet_name = f"{star_name} {planet_designators[j]}"
            
            if random.random() < 0.3: 
                sma = random.uniform(hz_inner * 0.8, hz_outer * 1.2)
            else:
                sma = 10 ** random.uniform(-1.5, 1.5)
            
            orbital_period = 365.25 * math.sqrt((sma ** 3) / star_mass)
            eccentricity = min(abs(random.gauss(0.05, 0.15)), 0.9)
            
            size_rand = random.random()
            if size_rand < 0.40:
                radius = random.uniform(0.5, 1.5)
            elif size_rand < 0.70:
                radius = random.uniform(1.5, 2.5)
            elif size_rand < 0.85:
                radius = random.uniform(2.5, 4)
            else: 
                radius = random. uniform(4, 15)
            
            if radius < 1.5:
                mass = radius ** 3.5
            elif radius < 4:
                mass = radius ** 2.5
            else:
                mass = 30 * (radius / 4) ** 2
            mass *= random.uniform(0.7, 1.5)
            
            star_radius = star_mass ** 0.8
            eq_temp = temp_k * ((1 - 0.3) ** 0.25) * math.sqrt(star_radius * 6.957e8 / (2 * sma * 1.496e11))
            
            surface_gravity = mass / (radius ** 2)
            density = (mass / (radius ** 3)) * 5.5
            planet_type = classify_planet_type(mass, radius, eq_temp)
            
            discovery_date = datetime. now() - timedelta(days=random. randint(0, 30 * 365))
            
            try:
                cursor. execute('''
                    INSERT INTO EXOPLANETS (star_id, name, mass_earth, radius_earth, 
                        orbital_period_days, semi_major_axis_au, eccentricity, equilibrium_temp_k,
                        surface_gravity_earth, density_gcc, planet_type, discovery_date, confirmed)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (star_id, planet_name, round(mass, 3), round(radius, 3),
                      round(orbital_period, 2), round(sma, 4), round(eccentricity, 4),
                      round(eq_temp, 1), round(surface_gravity, 3), round(density, 3),
                      planet_type, discovery_date. strftime('%Y-%m-%d'), 1))
                planet_ids.append(cursor.lastrowid)
            except: 
                continue
    
    # Generate habitability scores
    for planet_id in planet_ids:
        cursor.execute('''
            SELECT e.mass_earth, e. radius_earth, e.equilibrium_temp_k, e.semi_major_axis_au,
                   s.luminosity, s.temperature
            FROM EXOPLANETS e
            JOIN STAR_SYSTEMS s ON e. star_id = s.star_id
            WHERE e.planet_id = ? 
        ''', (planet_id,))
        
        data = cursor.fetchone()
        if not data:
            continue
            
        mass, radius, temp, sma, lum, star_temp = data
        
        escape_vel = math.sqrt(mass / radius) if radius > 0 else 1
        esi = calculate_esi(radius, mass, temp, escape_vel)
        
        hz_inner, hz_outer = calculate_habitable_zone(lum, star_temp)
        if hz_inner <= sma <= hz_outer: 
            hz_status = "In HZ"
        elif hz_inner * 0.8 <= sma <= hz_outer * 1.2:
            hz_status = "Near HZ Edge"
        elif sma < hz_inner: 
            hz_status = "Too Hot"
        else: 
            hz_status = "Too Cold"
        
        if hz_status == "In HZ" and 200 < temp < 350:
            water_prob = min(95, 50 + esi * 50)
        elif "Near" in hz_status: 
            water_prob = min(60, 20 + esi * 40)
        else:
            water_prob = max(0, esi * 20 - 10)
        
        atm_rating = random.randint(3, 8)
        mag_prob = min(90, 10 + mass * 20) if mass < 3 else max(10, 50 - mass)
        tidal_prob = max(0, min(100, 80 - sma * 50))
        
        if esi >= 0.8 and hz_status == "In HZ":
            hab_class = "Class I:  Prime Candidate"
            priority = random.randint(1, 2)
        elif esi >= 0.6:
            hab_class = "Class II:  High Potential"
            priority = random.randint(2, 4)
        elif esi >= 0.4:
            hab_class = "Class III: Moderate Interest"
            priority = random.randint(4, 6)
        else:
            hab_class = "Class IV: Low Priority"
            priority = random.randint(6, 10)
        
        cursor.execute('''
            INSERT INTO HABITABILITY_SCORES (planet_id, esi_score, hz_status, water_probability,
                atmosphere_rating, magnetic_field_probability, tidal_lock_probability,
                habitability_class, study_priority)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (planet_id, round(esi, 4), hz_status, round(water_prob, 1),
              atm_rating, round(mag_prob, 1), round(tidal_prob, 1), hab_class, priority))
    
    # Generate missions
    missions = [
        ('Kepler', 'NASA', 'Space Telescope', 'Transit', '2009-03-07', 'Retired', 10),
        ('TESS', 'NASA', 'Space Telescope', 'Transit', '2018-04-18', 'Active', 9),
        ('JWST', 'NASA/ESA/CSA', 'Space Telescope', 'Direct Imaging', '2021-12-25', 'Active', 10),
        ('HARPS', 'ESO', 'Ground Observatory', 'Radial Velocity', '2003-02-01', 'Active', 9),
    ]
    
    for m in missions:
        cursor.execute('''
            INSERT OR IGNORE INTO DISCOVERY_MISSIONS (name, organization, mission_type, detection_method,
                launch_date, status, sensitivity_rating)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', m)
    
    conn.commit()

# ============================================================================
# QUERY FUNCTIONS
# ============================================================================

def get_database_stats(conn):
    """Get database statistics"""
    stats = {}
    for table in ['STAR_SYSTEMS', 'EXOPLANETS', 'HABITABILITY_SCORES']: 
        cursor = conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        stats[table] = cursor.fetchone()[0]
    return stats

def get_top_candidates(conn, limit=20):
    """Get top habitable candidates"""
    query = f'''
        SELECT
            e.name as planet_name,
            s.name as host_star,
            s.spectral_type,
            ROUND(s.distance_ly, 1) as distance_ly,
            e.planet_type,
            ROUND(e.radius_earth, 2) as radius,
            ROUND(e.mass_earth, 2) as mass,
            ROUND(e.equilibrium_temp_k, 0) as temp_k,
            h.hz_status,
            ROUND(h. esi_score, 4) as esi_score,
            ROUND(h. water_probability, 1) as water_prob,
            h.habitability_class
        FROM EXOPLANETS e
        JOIN STAR_SYSTEMS s ON e. star_id = s.star_id
        JOIN HABITABILITY_SCORES h ON e.planet_id = h.planet_id
        ORDER BY h.esi_score DESC
        LIMIT {limit}
    '''
    return pd.read_sql_query(query, conn)

def get_habitability_summary(conn):
    """Get habitability class summary"""
    query = '''
        SELECT
            h.habitability_class,
            COUNT(*) as planet_count,
            ROUND(AVG(h. esi_score), 4) as avg_esi,
            ROUND(AVG(h.water_probability), 1) as avg_water_prob
        FROM HABITABILITY_SCORES h
        GROUP BY h.habitability_class
        ORDER BY avg_esi DESC
    '''
    return pd. read_sql_query(query, conn)

def get_stellar_analysis(conn):
    """Analyze by stellar type"""
    query = '''
        SELECT
            SUBSTR(s.spectral_type, 1, 1) as spectral_class,
            COUNT(DISTINCT s.star_id) as num_stars,
            COUNT(e.planet_id) as total_planets,
            SUM(CASE WHEN h.hz_status = 'In HZ' THEN 1 ELSE 0 END) as planets_in_hz,
            ROUND(AVG(h. esi_score), 4) as avg_esi
        FROM STAR_SYSTEMS s
        LEFT JOIN EXOPLANETS e ON s. star_id = e.star_id
        LEFT JOIN HABITABILITY_SCORES h ON e.planet_id = h.planet_id
        GROUP BY SUBSTR(s.spectral_type, 1, 1)
        ORDER BY avg_esi DESC
    '''
    return pd. read_sql_query(query, conn)

def search_planets(conn, min_esi=0, max_distance=1000, planet_type=None, hz_only=False):
    """Search planets with filters"""
    query = '''
        SELECT
            e.name as planet_name,
            s.name as host_star,
            s.spectral_type,
            ROUND(s. distance_ly, 1) as distance_ly,
            e.planet_type,
            ROUND(e.radius_earth, 2) as radius,
            ROUND(e.mass_earth, 2) as mass,
            ROUND(e.equilibrium_temp_k, 0) as temp_k,
            h. hz_status,
            ROUND(h.esi_score, 4) as esi,
            ROUND(h.water_probability, 1) as water_prob,
            h.habitability_class
        FROM EXOPLANETS e
        JOIN STAR_SYSTEMS s ON e.star_id = s.star_id
        JOIN HABITABILITY_SCORES h ON e. planet_id = h.planet_id
        WHERE h.esi_score >= ?  AND s.distance_ly <= ? 
    '''
    params = [min_esi, max_distance]
    
    if planet_type and planet_type != 'All':
        query += " AND e.planet_type LIKE ?"
        params.append(f"%{planet_type}%")
    
    if hz_only:
        query += " AND h. hz_status = 'In HZ'"
    
    query += " ORDER BY h.esi_score DESC LIMIT 100"
    
    return pd.read_sql_query(query, conn, params=params)

def get_planet_scatter_data(conn):
    """Get data for scatter plot"""
    query = '''
        SELECT
            e.name,
            e.mass_earth,
            e.radius_earth,
            e. equilibrium_temp_k,
            e.planet_type,
            h. esi_score,
            h.hz_status
        FROM EXOPLANETS e
        JOIN HABITABILITY_SCORES h ON e.planet_id = h.planet_id
    '''
    return pd.read_sql_query(query, conn)

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Optionally clear Streamlit cached resources (useful after deploy/code changes)
    if os.getenv('CLEAR_STREAMLIT_CACHE') == '1':
        try:
            st.cache_resource.clear()
        except Exception:
            pass

    # Initialize database
    conn = init_database()
    stats = get_database_stats(conn)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1 style="color: #00ff88; margin:  0;">🌌 GEDHAS</h1>
        <h3 style="color:  #66aaff; margin:  5px 0;">Galactic Exoplanet Discovery & Habitability Assessment System</h3>
        <p style="color: #aaaaaa; margin:  0;">Interactive Database Dashboard for Exoplanet Analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st. metric("🌟 Star Systems", stats['STAR_SYSTEMS'])
    with col2:
        st.metric("🪐 Exoplanets", stats['EXOPLANETS'])
    with col3:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM HABITABILITY_SCORES WHERE hz_status = 'In HZ'")
        hz_count = cursor. fetchone()[0]
        st.metric("🎯 In Habitable Zone", hz_count)
    with col4:
        cursor. execute("SELECT MAX(esi_score) FROM HABITABILITY_SCORES")
        max_esi = cursor.fetchone()[0]
        st.metric("📊 Highest ESI", f"{max_esi:.4f}")
    
    st.markdown("---")
    
    # Sidebar navigation
    st.sidebar.title("🛰️ Navigation")
    page = st.sidebar.radio(
        "Select View",
        ["🔍 Planet Search", "📊 Visualizations", "📋 Reports", "💻 Custom SQL"]
    )
    
    # Page content
    if page == "🔍 Planet Search":
        st.header("🔍 Exoplanet Search")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Search Filters")
            min_esi = st. slider("Minimum ESI Score", 0.0, 1.0, 0.0, 0.05)
            max_distance = st.slider("Maximum Distance (ly)", 10, 500, 200, 10)
            planet_type = st.selectbox(
                "Planet Type",
                ['All', 'Terrestrial', 'Super-Earth', 'Neptune', 'Jupiter']
            )
            hz_only = st.checkbox("Habitable Zone Only")
            
            if st.button("🔍 Search", type="primary"):
                results = search_planets(conn, min_esi, max_distance, planet_type, hz_only)
                st.session_state['search_results'] = results
        
        with col2:
            if 'search_results' in st.session_state:
                results = st.session_state['search_results']
                st.success(f"Found {len(results)} planets matching your criteria")
                st.dataframe(results, use_container_width=True)
            else:
                st.info("Configure filters and click Search to find planets")
    
    elif page == "📊 Visualizations":
        st.header("📊 Data Visualizations")
        
        viz_type = st.selectbox(
            "Select Visualization",
            ["Habitability Distribution", "Mass-Radius Diagram", "Stellar Type Analysis", "Discovery Timeline"]
        )
        
        if viz_type == "Habitability Distribution": 
            df = get_habitability_summary(conn)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.pie(df, values='planet_count', names='habitability_class',
                            title='Distribution of Habitability Classes',
                            color_discrete_sequence=px.colors.sequential.Plasma)
                st. plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.bar(df, x='habitability_class', y='avg_esi',
                            title='Average ESI by Habitability Class',
                            color='avg_esi', color_continuous_scale='RdYlGn')
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Mass-Radius Diagram": 
            df = get_planet_scatter_data(conn)
            
            fig = px.scatter(df, x='mass_earth', y='radius_earth',
                           color='esi_score', size='equilibrium_temp_k',
                           hover_name='name', hover_data=['planet_type', 'hz_status'],
                           title='Exoplanet Mass-Radius Diagram',
                           labels={'mass_earth': 'Mass (Earth masses)',
                                  'radius_earth':  'Radius (Earth radii)'},
                           color_continuous_scale='Plasma',
                           log_x=True, log_y=True)
            
            # Add Earth reference
            fig.add_hline(y=1, line_dash="dash", line_color="cyan", annotation_text="Earth radius")
            fig.add_vline(x=1, line_dash="dash", line_color="cyan", annotation_text="Earth mass")
            
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Stellar Type Analysis":
            df = get_stellar_analysis(conn)
            
            fig = make_subplots(rows=1, cols=2,
                               subplot_titles=('Planets per Stellar Type', 'Average ESI by Stellar Type'))
            
            fig.add_trace(
                go.Bar(x=df['spectral_class'], y=df['total_planets'], name='Total Planets',
                      marker_color='#66aaff'),
                row=1, col=1
            )
            fig.add_trace(
                go.Bar(x=df['spectral_class'], y=df['planets_in_hz'], name='In HZ',
                      marker_color='#00ff88'),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Bar(x=df['spectral_class'], y=df['avg_esi'], name='Avg ESI',
                      marker_color='#ff6688'),
                row=1, col=2
            )
            
            fig.update_layout(height=500, barmode='group')
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Discovery Timeline":
            query = '''
                SELECT strftime('%Y', discovery_date) as year, COUNT(*) as count
                FROM EXOPLANETS
                GROUP BY year ORDER BY year
            '''
            df = pd.read_sql_query(query, conn)
            df['cumulative'] = df['count'].cumsum()
            
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            fig.add_trace(
                go.Bar(x=df['year'], y=df['count'], name='Discoveries per Year',
                      marker_color='#66aaff'),
                secondary_y=False
            )
            fig.add_trace(
                go. Scatter(x=df['year'], y=df['cumulative'], name='Cumulative Total',
                          line=dict(color='#00ff88', width=3)),
                secondary_y=True
            )
            
            fig.update_layout(title='Exoplanet Discovery Timeline')
            fig.update_yaxes(title_text="Discoveries per Year", secondary_y=False)
            fig.update_yaxes(title_text="Cumulative Total", secondary_y=True)
            
            st.plotly_chart(fig, use_container_width=True)
    
    elif page == "📋 Reports":
        st. header("📋 Analysis Reports")
        
        report_type = st. selectbox(
            "Select Report",
            ["Top Habitable Candidates", "Habitability Summary", "Stellar Analysis", "Mission Overview"]
        )
        
        if report_type == "Top Habitable Candidates":
            limit = st.slider("Number of results", 10, 50, 20)
            df = get_top_candidates(conn, limit)
            st.subheader(f"🌍 Top {limit} Habitable Planet Candidates")
            st.dataframe(df, use_container_width=True)
            
            # Download button
            csv = df.to_csv(index=False)
            st.download_button("📥 Download CSV", csv, "top_candidates.csv", "text/csv")
        
        elif report_type == "Habitability Summary": 
            df = get_habitability_summary(conn)
            st.subheader("📊 Habitability Classification Summary")
            st.dataframe(df, use_container_width=True)
        
        elif report_type == "Stellar Analysis":
            df = get_stellar_analysis(conn)
            st.subheader("🌟 Analysis by Stellar Spectral Type")
            st.dataframe(df, use_container_width=True)
        
        elif report_type == "Mission Overview":
            df = pd.read_sql_query("SELECT * FROM DISCOVERY_MISSIONS", conn)
            st.subheader("🚀 Discovery Missions")
            st.dataframe(df, use_container_width=True)
    
    elif page == "💻 Custom SQL":
        st. header("💻 Custom SQL Query")
        
        st.info("""
        **Available Tables:**
        - `STAR_SYSTEMS` (star_id, name, spectral_type, luminosity, temperature, distance_ly, ...)
        - `EXOPLANETS` (planet_id, star_id, name, mass_earth, radius_earth, equilibrium_temp_k, planet_type, ...)
        - `HABITABILITY_SCORES` (planet_id, esi_score, hz_status, water_probability, habitability_class, ...)
        - `DISCOVERY_MISSIONS` (mission_id, name, organization, detection_method, status, ...)
        """)
        
        default_query = """SELECT 
    e.name as planet,
    s.name as star,
    ROUND(s.distance_ly, 1) as distance_ly,
    ROUND(h.esi_score, 4) as esi
FROM EXOPLANETS e
JOIN STAR_SYSTEMS s ON e.star_id = s.star_id
JOIN HABITABILITY_SCORES h ON e.planet_id = h. planet_id
WHERE h.esi_score > 0.5
ORDER BY h. esi_score DESC
LIMIT 20;"""
        
        query = st.text_area("Enter SQL Query", value=default_query, height=200)
        
        if st.button("▶️ Execute Query", type="primary"):
            try:
                if query.strip().upper().startswith('SELECT'):
                    df = pd.read_sql_query(query, conn)
                    st.success(f"Query returned {len(df)} rows")
                    st.dataframe(df, use_container_width=True)
                else: 
                    st.error("Only SELECT queries are allowed")
            except Exception as e: 
                st.error(f"Query Error: {e}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>🌌 GEDHAS - Galactic Exoplanet Discovery & Habitability Assessment System</p>
        <p>Built with Streamlit • Data is simulated for demonstration purposes</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__": 
    main()