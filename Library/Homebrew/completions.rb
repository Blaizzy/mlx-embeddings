# typed: true
# frozen_string_literal: true

require "utils/link"

# Helper functions for generating shell completions.
#
# @api private
module Completions
  extend T::Sig

  module_function

  sig { params(command: String).void }
  def link_if_allowed!(command: "brew completions link")
    if link_completions?
      link! command: command
    else
      unlink!
    end
  end

  sig { params(command: String).void }
  def link!(command: "brew completions link")
    write_completions_option "yes"
    Utils::Link.link_completions HOMEBREW_REPOSITORY, command
  end

  sig { void }
  def unlink!
    write_completions_option "no"
    Utils::Link.unlink_completions HOMEBREW_REPOSITORY
  end

  sig { returns(T::Boolean) }
  def link_completions?
    read_completions_option == "yes"
  end

  sig { returns(String) }
  def read_completions_option
    HOMEBREW_REPOSITORY.cd do
      Utils.popen_read("git", "config", "--get", "homebrew.linkcompletions").chomp
    end
  end

  sig { params(state: String).void }
  def write_completions_option(state)
    HOMEBREW_REPOSITORY.cd do
      T.unsafe(self).safe_system "git", "config", "--replace-all", "homebrew.linkcompletions", state.to_s
    end
  end
end
