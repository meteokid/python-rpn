/* -------------------------------------------- *
 *                                              *
 *  file      :  DECLARE.H                      *
 *                                              *
 *  author    :  Michel Grenier                 *
 *                                              *
 *  revision  :  V0.0                           *
 *                                              *
 *  status    :  DEVELOPMENT                    *
 *                                              *
 *  language  :  C                              *
 *                                              *
 *  os        :  UNIX, LINUX, WINDOS 95/98 NT   *
 *                                              *
 *  object    :  THIS FILE CONTAINS ALL THE     *
 *               DEFINITIONS NEEDED TO DECLARE  *
 *               C MODULES INTO C++             *
 *                                              *
 *  Copyright (C) 2019  Government of Canada
 *
 *  This library is free software; you can redistribute it and/or
 *  modify it under the terms of the GNU Lesser General Public
 *  License as published by the Free Software Foundation; version
 *  2.1 of the License.
 *
 *  This library is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *  Lesser General Public License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public
 *  License along with this library; if not, write to the Free
 *  Software Foundation, Inc., 51 Franklin Street, Fifth Floor,
 *  Boston, MA 02110-1301 USA
 *
 *  The Environment and Climate Change Canada open source licensing
 *  due diligence process form for libburpc was approved on December
 *  21, 2018.
 *
 * -------------------------------------------- */

#ifndef   include_DECLARE
#define   include_DECLARE

#ifndef   __BEGIN_DECLS

#    ifdef    __cplusplus
#      define __BEGIN_DECLS extern "C" {
#      define __END_DECLS              }
#    else
#      define __BEGIN_DECLS /* -vide- */
#      define __END_DECLS   /* -vide- */
#    endif /* __cplusplus  */

#endif /* __BEGIN_DECLS */

#endif /* include_DECLARE */
