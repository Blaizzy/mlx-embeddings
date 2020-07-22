# typed: true

module OS
  module Linux
    include Kernel

    def which(cmd, path = ENV["PATH"])
    end
  end

  module Mac
    include Kernel
  end
end

module OS::Mac
  class << self
    module Compat
      include Kernel
    end
  end
end
