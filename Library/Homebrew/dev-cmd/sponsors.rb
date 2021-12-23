# typed: false
# frozen_string_literal: true

require "cli/parser"
require "utils/github"

module Homebrew
  extend T::Sig

  module_function

  NAMED_TIER_AMOUNT = 100
  URL_TIER_AMOUNT = 1000

  sig { returns(CLI::Parser) }
  def sponsors_args
    Homebrew::CLI::Parser.new do
      description <<~EOS
        Update the list of GitHub Sponsors in the `Homebrew/brew` README.
      EOS

      named_args :none
    end
  end

  def sponsor_name(s)
    s["name"] || s["login"]
  end

  def sponsor_logo(s)
    "https://github.com/#{s["login"]}.png?size=64"
  end

  def sponsor_url(s)
    "https://github.com/#{s["login"]}"
  end

  def sponsors
    sponsors_args.parse

    named_sponsors = []
    logo_sponsors = []

    GitHub.sponsors_by_tier("Homebrew").each do |tier|
      if tier["tier"] >= NAMED_TIER_AMOUNT
        named_sponsors += tier["sponsors"].map do |s|
          "[#{sponsor_name(s)}](#{sponsor_url(s)})"
        end
      end

      next if tier["tier"] < URL_TIER_AMOUNT

      logo_sponsors += tier["sponsors"].map do |s|
        "[![#{sponsor_name(s)}](#{sponsor_logo(s)})](#{sponsor_url(s)})"
      end
    end

    named_sponsors << "many other users and organisations via [GitHub Sponsors](https://github.com/sponsors/Homebrew)"

    readme = HOMEBREW_REPOSITORY/"README.md"
    content = readme.read
    content.gsub!(/(Homebrew is generously supported by) .*\Z/m, "\\1 #{named_sponsors.to_sentence}.\n")
    content << "\n#{logo_sponsors.join}\n" if logo_sponsors.presence

    File.write(readme, content)

    diff = system_command "git", args: [
      "-C", HOMEBREW_REPOSITORY, "diff", "--exit-code", "README.md"
    ]
    if diff.status.success?
      puts "No changes to list of sponsors."
    else
      puts "List of sponsors updated in the README."
    end
  end
end
