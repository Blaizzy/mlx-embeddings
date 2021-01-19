# typed: false
# frozen_string_literal: true

require "cli/parser"
require "utils/github"

module Homebrew
  extend T::Sig

  module_function

  sig { returns(CLI::Parser) }
  def sponsors_args
    Homebrew::CLI::Parser.new do
      description <<~EOS
        Print a Markdown summary of Homebrew's GitHub Sponsors, suitable for pasting into a README.
      EOS

      named_args :none
    end
  end

  def sponsors
    sponsors_args.parse

    sponsors = {
      "named" => [],
      "users" => 0,
      "orgs"  => 0,
    }

    GitHub.sponsors_by_tier("Homebrew").each do |tier|
      sponsors["named"] += tier["sponsors"] if tier["tier"] >= 100
      sponsors["users"] += tier["count"]
      sponsors["orgs"] += tier["sponsors"].count { |s| s["type"] == "organization" }
    end

    items = []
    items += sponsors["named"].map { |s| "[#{s["name"]}](https://github.com/#{s["login"]})" }

    anon_users = sponsors["users"] - sponsors["named"].length - sponsors["orgs"]

    items << if items.length > 1
      "#{anon_users} other users"
    else
      "#{anon_users} users"
    end

    if sponsors["orgs"] == 1
      items << "#{sponsors["orgs"]} organization"
    elsif sponsors["orgs"] > 1
      items << "#{sponsors["orgs"]} organizations"
    end

    sponsor_text = if items.length > 2
      items[0..-2].join(", ") + " and #{items.last}"
    else
      items.join(" and ")
    end

    puts "Homebrew is generously supported by #{sponsor_text} via [GitHub Sponsors](https://github.com/sponsors/Homebrew)."
  end
end
