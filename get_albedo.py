#!/usr/bin/env python
import sqlite3
import numpy as np
import os
import sys
from collections import namedtuple

Results = namedtuple(
    "Res",
    "site year doy longitude latitude "
    + "bhr_vis bhr_nir albedo_qa snow_qa sun_angle",
)


def process_cli():
    """Process the command line
    """
    from optparse import OptionParser

    parser = OptionParser(
        usage="Jose Gomez-Dans "
        + "<j.gomez-dans@ucl.ac.uk>\n\n\tUsage: %prog [options]",
        version="%prog 1.0 Jose Gomez-dans <j.gomez-dans@ucl.ac.uk>",
    )

    parser.add_option(
        "-d",
        "--db",
        action="store",
        dest="db_file",
        default="albedo.sql",
        help="SQLite file to store albedo",
    )

    parser.add_option(
        "-l",
        "--longitude",
        action="store",
        default=None,
        dest="longitude",
        help="Longitude (decimal degrees)",
    )

    parser.add_option(
        "-t",
        "--latitude",
        action="store",
        default=None,
        dest="latitude",
        help="Latitude (decimal degrees)",
    )

    parser.add_option(
        "-y", "--year", action="store", dest="year", help="year"
    )

    parser.add_option(
        "-n",
        "--name",
        action="store",
        dest="name",
        help="The name of the site",
    )
    (options, args) = parser.parse_args()
    return (options, args)


def download_and_process_obs(longitude, latitude, year):
    """this function downloads the kernel weights for vis and nir, calculats
    white sky albedo from the kernel integrals in 
    `this page <http://www-modis.bu.edu/brdf/userguide/param.html>`_, and 
    also reads the QA data.
    
    .. todo::
    
       The QA data needs interpretation!
    """
    from modis_data.modis_data import get_modis_data

    # Of course, it could be far more terse, but this ought to work!
    year_start = year
    year_end = year
    (t_step, vis_par1) = get_modis_data(
        longitude,
        latitude,
        "MCD43A1",
        "BRDF_Albedo_Parameters_vis.Num_Parameters_01",
        year_start,
        year_end,
    )
    (t_step, vis_par2) = get_modis_data(
        longitude,
        latitude,
        "MCD43A1",
        "BRDF_Albedo_Parameters_vis.Num_Parameters_02",
        year_start,
        year_end,
    )
    (t_step, vis_par3) = get_modis_data(
        longitude,
        latitude,
        "MCD43A1",
        "BRDF_Albedo_Parameters_vis.Num_Parameters_03",
        year_start,
        year_end,
    )
    ws_albedo_vis = 1.0 * vis_par1 + 0.189184 * vis_par2 - 1.377622 * vis_par3
    (t_step, nir_par1) = get_modis_data(
        longitude,
        latitude,
        "MCD43A1",
        "BRDF_Albedo_Parameters_nir.Num_Parameters_01",
        year_start,
        year_end,
    )
    (t_step, nir_par2) = get_modis_data(
        longitude,
        latitude,
        "MCD43A1",
        "BRDF_Albedo_Parameters_nir.Num_Parameters_02",
        year_start,
        year_end,
    )
    (t_step, nir_par3) = get_modis_data(
        longitude,
        latitude,
        "MCD43A1",
        "BRDF_Albedo_Parameters_nir.Num_Parameters_03",
        year_start,
        year_end,
    )
    ws_albedo_nir = 1.0 * nir_par1 + 0.189184 * nir_par2 - 1.377622 * nir_par3

    (t_step, albedo_qa) = get_modis_data(
        longitude,
        latitude,
        "MCD43A2",
        "BRDF_Albedo_Ancillary",
        year_start,
        year_end,
    )
    sun_angle = []
    for qa_datum in albedo_qa:
        qax = int(qa_datum[0])
        sun_angle.append(float(int(bin(int(qax))[2:][1:8][::-1], 2)))
    sun_angle = np.array(sun_angle)
    (t_step, albedo_qa) = get_modis_data(
        longitude,
        latitude,
        "MCD43A2",
        "BRDF_Albedo_Quality",
        year_start,
        year_end,
    )

    (t_step, snow_qa) = get_modis_data(
        longitude,
        latitude,
        "MCD43A2",
        "Snow_BRDF_Albedo",
        year_start,
        year_end,
    )

    return (
        t_step,
        ws_albedo_vis,
        ws_albedo_nir,
        albedo_qa,
        snow_qa,
        sun_angle,
    )


class Observations(object):
    """An object that stores MODIS-type kernels observations in an SQLITE dB"""

    def __init__(self, db_file):
        """The class creator just creates the DB if it doesn't exist on the
        file system"""
        if not os.path.exists(db_file):
            self._create_albedo_db(db_file)

        self.db = sqlite3.connect(db_file)
        self.cursor = self.db.cursor()

    def query(self, year, site_name, longitude=None, latitude=None):
        """The query method is responsible for querying the DB. If it can't
        find the data it wants, it will download it, store it in the DB and 
        return it."""
        while True:
            result = self.cursor.execute(
                """
                SELECT * FROM albedo WHERE site='%s' AND year=%d;
                """
                % (site_name, year)
            )
            try:
                # We already have data stored ;-)
                first_row = next(result)
                t_axis = []
                albedo_vis = []
                albedo_nir = []
                qa = []
                snow = []
                sun = []
                for row in [first_row] + result.fetchall():
                    t_axis.append(row[4])
                    albedo_vis.append(row[5])
                    albedo_nir.append(row[6])
                    qa.append(row[7])
                    snow.append(row[8])
                    sun.append(row[9])
                results = Results(
                    site=site_name,
                    year=year,
                    longitude=longitude,
                    latitude=latitude,
                    doy=np.array(t_axis),
                    bhr_vis=np.array(albedo_vis),
                    bhr_nir=np.array(albedo_nir),
                    albedo_qa=np.array(qa),
                    snow_qa=np.array(snow),
                    sun_angle=np.array(sun),
                )
                return results

            except StopIteration as e:
                # No data for this site and year
                assert (
                    longitude is not None
                ), "Longitude must be defined to dload data"
                assert (
                    latitude is not None
                ), "Latitude must be defined to dload data"
                longitude = float(longitude)
                latitude = float(latitude)
                (
                    t_step,
                    ws_albedo_vis,
                    ws_albedo_nir,
                    albedo_qa,
                    snow_qa,
                    sun_angle,
                ) = download_and_process_obs(longitude, latitude, year)
                data = []
                for i, t in enumerate(t_step):
                    tup = (
                        site_name,
                        longitude,
                        latitude,
                        year,
                        t - year * 1000,
                        float(ws_albedo_vis[i]) / 1000.0,
                        float(ws_albedo_nir[i]) / 1000.0,
                        int(albedo_qa[i]),
                        int(snow_qa[i]),
                        float(sun_angle[i]),
                    )
                    data.append(tup)

                self.cursor.executemany(
                    """
                        INSERT INTO albedo(site, longitude, latitude, year, time, bhr_vis, bhr_nir, albedo_qa, snow_qa, sun_angle) VALUES (?,?,?,?,?,?,?,?,?,?)
                        """,
                    data,
                )
                self.db.commit()
        self.db.close()

    def _create_albedo_db(self, db_file):
        """Creates the DB in case it didn't exist already"""
        db = sqlite3.connect(db_file)
        cursor = db.cursor()
        result = cursor.execute(
            """
            CREATE TABLE albedo(site TEXT, longitude REAL, latitude REAL,
            year INTEGER, time INTEGER,
            bhr_vis REAL, bhr_nir REAL, albedo_qa INTEGER, snow_qa INTEGER,
            sun_angle REAL) ;
            """
        )
        db.commit()
        db.close()


if __name__ == "__main__":

    # ( options, args ) = process_cli ()
    obs = Observations("albedo.sql")
    with open("fluxnet_sites_set.txt", "r") as fp:
        for line in fp:
            if not line.startswith("#"):
                (site, site_code, nyears, lat, lon, lc1, lc2) = line.split(
                    ";"
                )
                print("Grabbing %s (%s)->(%s)" % (site_code, lc1, lc2))
                for year in range(2004, 2014):
                    print("\tYear %d" % year)
                    try:
                        x = obs.query(
                            year,
                            site_code.replace("'", ""),
                            longitude=float(lon),
                            latitude=float(lat),
                        )
                    except:
                        print("It seems I can't do it")
