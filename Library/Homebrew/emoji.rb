# frozen_string_literal: true

module Emoji
  class << self
    def install_badge
      ENV["HOMEBREW_INSTALL_BADGE"] || "🍺"
    end

    def enabled?
      !ENV["HOMEBREW_NO_EMOJI"]
    end
  end
end
