# typed: strict

class Formula
  module Compat
    include Kernel

    def any_installed_keg; end

    def latest_installed_prefix; end
  end
end
