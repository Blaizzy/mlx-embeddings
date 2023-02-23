# typed: false
# frozen_string_literal: true

module Homebrew
  extend T::Sig

  module_function

  class << self
    alias generic_git_tags git_tags
  end

  def git_tags
    tags = generic_git_tags
    tags = Utils.popen_read("git tag --list | sort -rV") if tags.blank?
    tags
  end
end
