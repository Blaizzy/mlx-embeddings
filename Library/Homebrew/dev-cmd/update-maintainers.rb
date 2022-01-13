# typed: false
# frozen_string_literal: true

require "cli/parser"
require "utils/github"
require "dev-cmd/generate-man-completions"

module Homebrew
  extend T::Sig

  module_function

  sig { returns(CLI::Parser) }
  def update_maintainers_args
    Homebrew::CLI::Parser.new do
      description <<~EOS
        Update the list of maintainers in the `Homebrew/brew` README.
      EOS

      named_args :none
    end
  end

  def update_maintainers
    update_maintainers_args.parse

    # We assume that only public members wish to be included in the README
    public_members = GitHub.public_member_usernames("Homebrew")

    members = {
      plc: GitHub.members_by_team("Homebrew", "plc"),
      tsc: GitHub.members_by_team("Homebrew", "tsc"),
    }
    members[:other] = GitHub.members_by_team("Homebrew", "maintainers")
                            .except(*members.values.map(&:keys).flatten.uniq)

    sentences = {}
    members.each do |group, hash|
      hash.slice!(*public_members)
      hash.each { |login, name| hash[login] = "[#{name}](https://github.com/#{login})" }
      sentences[group] = hash.values.sort.to_sentence
    end

    readme = HOMEBREW_REPOSITORY/"README.md"

    content = readme.read
    content.gsub!(/(Homebrew's \[Project Leadership Committee.*) is .*\./,
                  "\\1 is #{sentences[:plc]}.")
    content.gsub!(/(Homebrew's \[Technical Steering Committee.*) is .*\./,
                  "\\1 is #{sentences[:tsc]}.")
    content.gsub!(/(Homebrew's other current maintainers are).*\./,
                  "\\1 #{sentences[:other]}.")

    File.write(readme, content)

    diff = system_command "git", args: [
      "-C", HOMEBREW_REPOSITORY, "diff", "--exit-code", "README.md"
    ]
    if diff.status.success?
      puts "No changes to list of maintainers."
    else
      Homebrew.regenerate_man_pages(preserve_date: true, quiet: true)
      puts "List of maintainers updated in the README and the generated man pages."
    end
  end
end
