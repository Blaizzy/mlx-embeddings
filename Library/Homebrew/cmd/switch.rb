# frozen_string_literal: true

require "formula"
require "keg"
require "cli/parser"

module Homebrew
  module_function

  def switch_args
    Homebrew::CLI::Parser.new do
      usage_banner <<~EOS
        `switch` <formula> <version>

        Symlink all of the specified <version> of <formula>'s installation into Homebrew's prefix.
      EOS

      named 2
    end
  end

  def switch
    args = switch_args.parse

    name = args.named.first
    rack = Formulary.to_rack(name)

    odie "#{name} not found in the Cellar." unless rack.directory?

    versions = rack.subdirs
                   .map { |d| Keg.new(d).version }
                   .sort
                   .join(", ")
    version = args.named.second

    odie <<~EOS unless (rack/version).directory?
      #{name} does not have a version \"#{version}\" in the Cellar.
      #{name}'s installed versions: #{versions}
    EOS

    # Unlink all existing versions
    rack.subdirs.each do |v|
      keg = Keg.new(v)
      puts "Cleaning #{keg}"
      keg.unlink
    end

    keg = Keg.new(rack/version)

    # Link new version, if not keg-only
    if Formulary.keg_only?(rack)
      keg.optlink(verbose: args.verbose?)
      puts "Opt link created for #{keg}"
    else
      puts "#{keg.link} links created for #{keg}"
    end
  end
end
