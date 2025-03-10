source "https://rubygems.org"

# Use GitHub Pages dependencies
gem "github-pages", group: :jekyll_plugins

# This is the default theme for new Jekyll sites
gem "minima", "~> 2.5"

# Jekyll plugins
group :jekyll_plugins do
  gem "jekyll-feed", "~> 0.12"
end

# Windows-specific gems
platforms :mingw, :x64_mingw, :mswin, :jruby do
  gem "tzinfo", ">= 1", "< 3"
  gem "tzinfo-data"
end

# Performance booster for watching directories on Windows
gem "wdm", "~> 0.1", platforms: [:mingw, :x64_mingw, :mswin]

# Fix for JRuby
gem "http_parser.rb", "~> 0.6.0", platforms: [:jruby]
