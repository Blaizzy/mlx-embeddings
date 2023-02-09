# typed: false
# frozen_string_literal: true

module Homebrew
  extend T::Sig

  module_function

  def git_tag
    tags = generic_git_tag
    Utils.popen_read("git tag --list | sort -rV") if tags.blank?
  end
end
