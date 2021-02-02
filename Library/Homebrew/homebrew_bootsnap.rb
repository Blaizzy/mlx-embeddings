# typed: ignore
# frozen_string_literal: true

# TODO: make this `typed: true` when HOMEBREW_BOOTSNAP is enabled by
# default and/or we vendor bootsnap and the RBI file.
if !ENV["HOMEBREW_NO_BOOTSNAP"] &&
   ENV["HOMEBREW_BOOTSNAP"] &&
   # portable ruby doesn't play nice with bootsnap
   !ENV["HOMEBREW_FORCE_VENDOR_RUBY"] &&
   (!ENV["HOMEBREW_MACOS_VERSION"] || ENV["HOMEBREW_MACOS_SYSTEM_RUBY_NEW_ENOUGH"])

  require "rubygems"

  begin
    require "bootsnap"
  rescue LoadError
    raise if ENV["HOMEBREW_BOOTSNAP_RETRY"]

    Dir.chdir(HOMEBREW_LIBRARY_PATH) do
      system "bundle", "install", "--standalone"
    end

    ENV["HOMEBREW_BOOTSNAP_RETRY"] = "1"
    exec ENV["HOMEBREW_BREW_FILE"], *ARGV
  end

  ENV.delete("HOMEBREW_BOOTSNAP_RETRY")

  cache = ENV["HOMEBREW_CACHE"] || ENV["HOMEBREW_DEFAULT_CACHE"]
  # Can't use .blank? here because we haven't required active_support yet.
  raise "Needs HOMEBREW_CACHE or HOMEBREW_DEFAULT_CACHE!" if cache.nil? || cache.empty? # rubocop:disable Rails/Blank

  Bootsnap.setup(
    cache_dir:          cache,
    load_path_cache:    true,
    compile_cache_iseq: true,
    compile_cache_yaml: true,
  )
end
