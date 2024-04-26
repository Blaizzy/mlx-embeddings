# typed: strict

module SharedEnvExtension
  include EnvMethods
end

# @!visibility private
class Sorbet
  module Private
    module Static
      class ENVClass
        include SharedEnvExtension
      end
    end
  end
end
