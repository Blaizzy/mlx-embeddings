# frozen_string_literal: true

require "cask/staged"

module Cask
  class DSL
    class UninstallPreflight < Base
      include Staged
    end
  end
end
