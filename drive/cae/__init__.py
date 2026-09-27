"""drive/cae -- the AeroBO-look widget kit the garage's MISSION and DESIGN
pages are drawn with: AeroBO v3's light CAE desktop (hairlines and title
bars, one desk-blue accent, the system UI face with a monospace face for
every number) rebuilt in pygame.

Pure UI: no physics, no file IO beyond the bundled icon font. The shell
around the pages is drive/design_shell.py; the stage views are drive/views_*.

Modules
    theme     tokens, sizes (set_scale), fonts and icons, cached text, tint /
              fade, ellipsis, the hint split rule, number formatting
    form      Form, the ParamList-compatible object a shell view binds its
              controls to
    widgets   WorkUI, the immediate-mode work area, and Overlay
    plot      Figure, plotly-look 2-D plots
    chrome    menu bar, tool bar, tree, properties, tabs, output log, status
              bar, toasts, dialogs

Re-exports nothing: import the module you need, and read theme constants as
`T.NAME` at use time (`from drive.cae import theme as T`) -- `T.set_scale`
rewrites the sizes in place, so a copied name would go stale.
"""
