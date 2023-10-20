# hugo starter site

Created with help from Sebastian Pech's list of [Hugo best practices](https://github.com/spech66/hugo-best-practices).

I build lots of Hugo sites, around 3 sites per month. And I don't like the default site layout that Hugo creates when you run `hugo new site`. It doesn't have all the necessary elements to get started with, and has some you don't really need. 

Most importantly, it's useful for a site to have placeholder posts to work with. The goal is to be able to copy the folder, open it up, and start working on design immediately. without having to worry about folder structure, necessary files, and where my stylesheets and fonts go. 


## Features:
- Header and footer partials
- Templates for single, list, taxonomy pages, that use the `baseof` template correctly. 
- CSS, JS, and fonts all stored within the `static` directory
- Two different kinds of posts, to test different content formats
- Dynamic social media/preview cards
- Deafult archetype with all the essential front-matter
- 404 page
- RSS feed (index.xml)


## Missing features:

- **Assets:** If you use Hugo's `assets` folder and build pipelines, you can set up things like automatic image optimization and minification or stylesheets/scripts. But I don't like thinking about that stuff when I'm developing, I want the fewest possible number of folders, and the `static` folder is the more obvious default to put all your static files into. 
- **Data:** This folder usually contains a bunch of TOML/YAML/JSON files that are used to add content to certain parts of the site, without creating a separate page for them. Often unnecessary for blog-based sites. Also really simple to add if you need them.  
- **Thumbnails:** Hugo recommends putting thumbnail images (as well as other media files) in the same folder as a post (yes, it also recommends creating a folder for each post), which I think is way too cumbersome. Instead I just store all  images in the `static/media` folder and add sub-folders when necessary. 
