# typed: true
# frozen_string_literal: true

require "utils/link"

# Helper functions for generating shell completions.
#
# @api private
module Completions
  extend T::Sig

  module_function

  sig { void }
  def link!
    write_completions_option "yes"
    Tap.each do |tap|
      Utils::Link.link_completions tap.path, "brew completions link"
    end
  end

  sig { void }
  def unlink!
    write_completions_option "no"
    Tap.each do |tap|
      next if tap.official?

      Utils::Link.unlink_completions tap.path
    end
  end

  sig { returns(T::Boolean) }
  def link_completions?
    read_completions_option == "yes"
  end

  sig { returns(T::Boolean) }
  def completions_to_link?
    shells = %w[bash fish zsh]
    Tap.each do |tap|
      next if tap.official?

      shells.each do |shell|
        return true if (tap.path/"completions/#{shell}").exist?
      end
    end

    false
  end

  sig { params(option: String).returns(String) }
  def read_completions_option(option: "linkcompletions")
    HOMEBREW_REPOSITORY.cd do
      Utils.popen_read("git", "config", "--get", "homebrew.#{option}").chomp
    end
  end

  sig { params(state: String, option: String).void }
  def write_completions_option(state, option: "linkcompletions")
    HOMEBREW_REPOSITORY.cd do
      T.unsafe(self).safe_system "git", "config", "--replace-all", "homebrew.#{option}", state.to_s
    end
  end

  sig { void }
  def show_completions_message_if_needed
    return if read_completions_option(option: "completionsmessageshown") == "yes"
    return unless completions_to_link?

    T.unsafe(self).ohai "Homebrew completions for external commands are unlinked by default!"
    T.unsafe(self).puts <<~EOS
      To opt-in to automatically linking Homebrew shell competion files, run:
        brew completions link
      Then, follow the directions at #{Formatter.url("https://docs.brew.sh/Shell-Completion")}
    EOS

    write_completions_option("yes", option: "completionsmessageshown")
  end
end
