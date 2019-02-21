# Never `require` anything in this file (except English). It needs to be able to
# work as the first item in `brew.rb` so we can load gems with Bundler when
# needed before anything else is loaded (e.g. `json`).

require "English"

module Homebrew
  module_function

  def ruby_bindir
    "#{RbConfig::CONFIG["prefix"]}/bin"
  end

  def setup_gem_environment!
    # Match where our bundler gems are.
    ENV["GEM_HOME"] = "#{ENV["HOMEBREW_LIBRARY"]}/Homebrew/vendor/bundle/ruby/#{RbConfig::CONFIG["ruby_version"]}"
    ENV["GEM_PATH"] = ENV["GEM_HOME"]

    # Make RubyGems notice environment changes.
    Gem.clear_paths
    Gem::Specification.reset

    # Add necessary Ruby and Gem binary directories to PATH.
    paths = ENV["PATH"].split(":")
    paths.unshift(ruby_bindir) unless paths.include?(ruby_bindir)
    paths.unshift(Gem.bindir) unless paths.include?(Gem.bindir)
    ENV["PATH"] = paths.compact.join(":")
  end

  def install_gem!(name, version = nil)
    setup_gem_environment!
    return unless Gem::Specification.find_all_by_name(name, version).empty?

    # Shell out to `gem` to avoid RubyGems requires e.g. loading JSON.
    puts "==> Installing '#{name}' gem"
    install_args = %W[--no-document #{name}]
    install_args << "--version" << version if version
    return if system "#{ruby_bindir}/gem", "install", *install_args

    $stderr.puts "Error: failed to install the '#{name}' gem."
    exit 1
  end

  def install_gem_setup_path!(name, executable: name)
    install_gem!(name)
    return if ENV["PATH"].split(":").any? do |path|
      File.executable?("#{path}/#{executable}")
    end

    $stderr.puts <<~EOS
      Error: the '#{name}' gem is installed but couldn't find '#{executable}' in the PATH:
      #{ENV["PATH"]}
    EOS
    exit 1
  end

  def install_bundler!
    install_gem_setup_path! "bundler", executable: "bundle"
  end

  def install_bundler_gems!
    install_bundler!

    ENV["BUNDLE_GEMFILE"] = "#{ENV["HOMEBREW_LIBRARY"]}/Homebrew/test/Gemfile"
    @bundle_installed ||= begin
      bundle_check_output = `#{Gem.bindir}/bundle check`
      bundle_check_failed = !$CHILD_STATUS.exitstatus.zero?

      # for some reason sometimes the exit code lies so check the output too.
      if bundle_check_failed || bundle_check_output.include?("Install missing gems")
        system "#{Gem.bindir}/bundle", "install"
      else
        true
      end
    end

    setup_gem_environment!
  end
end
